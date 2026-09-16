"""Stop a LEGO Education car in the middle of the webcam's view.

An AprilTag (tag36h11, id 0) is taped to a tower on the car. The webcam
watches the car drive back and forth parallel to the screen; this script
finds the tag, works out how far its centre sits from the centre of the
frame, and drives the car until that error is ~0 -- then brakes.

Two refinements on a plain P controller: the apparent size of the tag tells
us how far away the car is, and the gains scale with it (far = fast, close =
slow), and the controller steers on where the car *will* be once the next
Bluetooth command lands rather than where it is now, which is what stops it
hunting back and forth across the centre line.

It connects to the Double Motor on the orange Connection Card, serial 1129.

    source .venv/bin/activate
    python apriltagparking.py                 # connect to the car and park it
    python apriltagparking.py --no-robot      # vision only, no Bluetooth

Keys while the preview window has focus:
    f       flip the drive direction (if the car runs away from centre)
    space   re-arm: leave PARKED and start hunting for the centre again
    q/ESC   quit (motors are always stopped on the way out)
"""

import argparse
import sys
import time

import cv2
import numpy as np

import legoeducation as le

# --- hardware --------------------------------------------------------------
# Our car's Double Motor: the Connection Card is orange, serial 1129.
CARD_COLOR = le.LEGO_COLOR_ORANGE
CARD_SERIAL = "1129"        # a string, so leading zeros survive

# --- tuning ----------------------------------------------------------------
TAG_ID = 0                  # the id printed on the tag taped to the car
TAG_DICT = cv2.aruco.DICT_APRILTAG_36H11

KP = 90.0                   # speed (%) per unit of normalised error
MAX_SPEED = 45              # base cap, before the distance scaling
MIN_SPEED = 14              # base floor: below this the car stalls
SPEED_CEILING = 70          # absolute cap, however far away the car is
SPEED_FLOOR = 9             # absolute floor, however close it is
PATROL_SPEED = 30           # speed of the back-and-forth search sweep

# Distance scaling. The tag is a known 100 mm square, so its apparent width
# in the frame is an inverse proxy for range: small tag = far away. Gains are
# multiplied by (reference span / measured span), so the car drives hard when
# it is across the room and creeps when it is close. This needs no camera
# calibration -- it is a ratio of two pixel measurements.
DEPTH_REF_SPAN = 0.10       # tag width (fraction of frame) where gain == 1
DEPTH_GAIN_MIN = 0.55
DEPTH_GAIN_MAX = 1.9

# Lead compensation. A frame takes ~30 ms and the BLE command another ~80 ms,
# so by the time a brake lands the car has already moved. Steering on the
# error projected LEAD_S into the future makes it start braking early instead
# of sailing through the centre and correcting back -- the cause of the
# oscillation.
LEAD_S = 0.25               # seconds of lead
RATE_SMOOTH = 0.6           # EMA weight on the measured error rate (0..1)
STILL_RATE = 0.06           # |error rate| under this counts as stopped

TOLERANCE = 0.02            # |error| under this (fraction of width) = centred
RELEASE = 0.06              # ...and over this, a parked car starts driving again
HOLD_FRAMES = 4             # consecutive centred frames required to park

PATROL_FLIP_S = 2.5         # reverse the sweep after this long with no tag

LOST_GRACE_S = 0.4          # keep the last command this long before giving up
COMMAND_PERIOD_S = 0.08     # BLE rate limit: at most ~12 commands a second
SPEED_EPSILON = 3           # don't resend a speed that barely changed

TAG_MM = 100.0              # printed tag size, for the HUD distance readout
CAMERA_HFOV_DEG = 60.0      # rough webcam field of view -- HUD only, not control


def _color_name(color):
    return le.LEGO_COLOR_NAME_MAP[color].removeprefix("LEGO_COLOR_").lower()


class Car:
    """The Double Motor, wrapped so --no-robot can stand in for hardware."""

    def __init__(self, enabled=True, card_color=CARD_COLOR,
                 card_serial=CARD_SERIAL):
        self.enabled = enabled
        self.card_color = card_color
        self.card_serial = card_serial
        self.motor = None
        self._last_speed = None
        self._last_sent = 0.0
        self._last_light = None

    def connect(self):
        if not self.enabled:
            print("[car] --no-robot: running vision only")
            return
        print(f"[car] scanning for the Double Motor on Connection Card "
              f"{_color_name(self.card_color)} {self.card_serial}...")
        self.motor = le.DoubleMotor()
        self.motor.connect(card_color=self.card_color,
                           card_serial=self.card_serial)
        self.motor.movement_set_end_state(le.MOTOR_END_STATE_BRAKE)
        print("[car] connected")

    def drive(self, speed):
        """Drive both sides at `speed` (-100..100), rate limited."""
        speed = int(np.clip(speed, -100, 100))
        now = time.monotonic()
        settled = speed == 0 and self._last_speed == 0
        close = (self._last_speed is not None
                 and abs(speed - self._last_speed) < SPEED_EPSILON
                 and speed != 0)
        if settled or close or now - self._last_sent < COMMAND_PERIOD_S:
            return
        self._last_speed = speed
        self._last_sent = now
        if self.motor is None:
            return
        if speed == 0:
            self.motor.movement_stop(blocking=False)
        else:
            self.motor.movement_move_tank(speed, speed, blocking=False)

    def stop(self):
        self._last_speed, self._last_sent = 0, 0.0
        if self.motor is not None:
            self.motor.movement_stop()

    def signal(self, parked):
        """Green light once parked, orange while hunting -- only on change."""
        if parked == self._last_light:
            return
        self._last_light = parked
        if self.motor is not None:
            self.motor.light_color(
                le.LEGO_COLOR_GREEN if parked else le.LEGO_COLOR_ORANGE,
                blocking=False)

    def close(self):
        if self.motor is not None:
            self.stop()
            self.motor.disconnect()
            print("[car] disconnected")


def make_detector():
    params = cv2.aruco.DetectorParameters()
    # sub-pixel corners: the tag centre is what we steer on, so it pays
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(TAG_DICT),
                                   params)


def find_tag(detector, frame, tag_id):
    """Return (centre_xy, corners) for `tag_id`, or (None, None)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None, None
    for quad, found in zip(corners, ids.flatten()):
        if found == tag_id:
            pts = quad.reshape(4, 2)
            return pts.mean(axis=0), pts
    return None, None


RED = (0, 0, 255)           # BGR
HUD_BAND = 70               # px of status text at the top of the frame


def tag_span(quad):
    """Mean side length of the tag in pixels -- our stand-in for range."""
    sides = [np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)]
    return float(np.mean(sides))


def depth_gain(span_px, width_px):
    """Gain multiplier from apparent tag size: far away = big, close = small."""
    span = max(span_px / width_px, 1e-6)
    return float(np.clip(DEPTH_REF_SPAN / span, DEPTH_GAIN_MIN, DEPTH_GAIN_MAX))


def approx_distance_mm(span_px, width_px):
    """Rough range for the HUD only, from an assumed field of view."""
    focal_px = (width_px / 2.0) / np.tan(np.radians(CAMERA_HFOV_DEG / 2.0))
    return TAG_MM * focal_px / max(span_px, 1e-6)


def draw_hud(frame, centre, quad, error, state, speed, sign, lead, gain, dist):
    h, w = frame.shape[:2]
    mid = w // 2
    band = int(TOLERANCE * w)

    cv2.line(frame, (mid, 0), (mid, h), (0, 255, 255), 1)
    cv2.rectangle(frame, (mid - band, 0), (mid + band, h - 1), (0, 200, 200), 1)

    if quad is not None:
        cv2.polylines(frame, [quad.astype(np.int32)], True, RED, 2)
        cx, cy = int(round(centre[0])), int(round(centre[1]))
        cv2.drawMarker(frame, (cx, cy), RED, cv2.MARKER_CROSS, 18, 2)
        cv2.circle(frame, (cx, cy), 4, RED, -1)
        # how far the centroid sits from the centre line, drawn to scale
        cv2.line(frame, (mid, cy), (cx, cy), RED, 1, cv2.LINE_AA)
        # Label the centroid above the tag, on a filled plate: red text over
        # the tag's own black squares is unreadable, and the plate keeps it
        # legible against whatever the tag happens to be sitting on.
        label = f"({cx}, {cy})"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        tx = int(np.clip(cx - tw // 2, 4, w - tw - 4))
        above = int(quad[:, 1].min()) - 10
        # ...unless that would land it on the status lines, in which case the
        # label goes under the tag instead.
        ty = above if above - th > HUD_BAND else int(quad[:, 1].max()) + th + 10
        ty = int(np.clip(ty, th + 6, h - base - 4))
        cv2.rectangle(frame, (tx - 5, ty - th - 5), (tx + tw + 5, ty + base + 3),
                      (255, 255, 255), -1)
        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    RED, 2, cv2.LINE_AA)

    colour = (0, 255, 0) if state == "PARKED" else (255, 255, 255)
    err = "  --" if error is None else f"{error:+.3f}"
    cv2.putText(frame, f"{state}  err={err}  speed={speed:+d}  dir={sign:+d}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    if error is not None:
        cv2.putText(frame,
                    f"lead={lead:+.3f}  gain={gain:.2f}x  ~{dist / 10.0:.0f} cm",
                    (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 120), 2)
    cv2.putText(frame, "f flip dir   space re-arm   q quit",
                (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


def park(cap, car, detector, tag_id, sign, mirror):
    state = "SEARCH"            # SEARCH -> TRACK -> PARKED
    centred_frames = 0
    patrol_dir = 1
    patrol_since = time.monotonic()
    last_seen = 0.0
    speed = 0

    rate = 0.0                  # d(error)/dt, smoothed
    prev_error = prev_t = None
    gain, lead, dist = 1.0, 0.0, 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[cam] dropped frame", file=sys.stderr)
            break
        if mirror:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        centre, quad = find_tag(detector, frame, tag_id)
        now = time.monotonic()

        error = None
        if centre is not None:
            # normalised: -0.5 = far left edge, 0 = centred, +0.5 = far right
            error = (centre[0] - w / 2.0) / w
            span = tag_span(quad)
            gain = depth_gain(span, w)
            dist = approx_distance_mm(span, w)

            if prev_error is not None:
                dt = now - prev_t
                if dt > 1e-3:
                    measured = (error - prev_error) / dt
                    rate = RATE_SMOOTH * rate + (1.0 - RATE_SMOOTH) * measured
            prev_error, prev_t = error, now
            last_seen = now

            # Where the car will be once this command actually takes effect.
            lead = error + LEAD_S * rate
        else:
            # Re-acquiring after a dropout must not read as a huge jump.
            prev_error = prev_t = None
            rate = 0.0

        if state == "PARKED":
            speed = 0
            if error is not None and abs(error) > RELEASE:
                state, centred_frames = "TRACK", 0

        elif error is not None:
            state = "TRACK"
            if abs(lead) <= TOLERANCE:
                # On target, or heading there fast enough that braking now
                # lands it there. Either way: stop driving.
                speed = 0
                if abs(error) <= TOLERANCE and abs(rate) <= STILL_RATE:
                    centred_frames += 1
                    if centred_frames >= HOLD_FRAMES:
                        state = "PARKED"
                        print(f"[park] centred (err={error:+.3f}, "
                              f"~{dist / 10.0:.0f} cm) -- braking")
                else:
                    centred_frames = 0
            else:
                centred_frames = 0
                # Far away, a pixel of error is many millimetres of floor, so
                # `gain` scales both the response and its limits with range.
                floor = max(MIN_SPEED * gain, SPEED_FLOOR)
                ceiling = min(MAX_SPEED * gain, SPEED_CEILING)
                magnitude = float(np.clip(abs(lead) * KP * gain, floor, ceiling))
                # lead > 0 means the car is (headed) right of centre, so drive
                # the way that shrinks it; `sign` absorbs the car's orientation.
                speed = int(round(-np.sign(lead) * magnitude * sign))

        elif now - last_seen < LOST_GRACE_S and state == "TRACK":
            pass                                  # brief dropout: coast on

        else:
            # No tag: sweep back and forth until the car drives into view.
            state, centred_frames = "SEARCH", 0
            if now - patrol_since > PATROL_FLIP_S:
                patrol_dir *= -1
                patrol_since = now
            speed = PATROL_SPEED * patrol_dir * sign

        car.drive(speed)
        car.signal(state == "PARKED")

        draw_hud(frame, centre, quad, error, state, speed, sign, lead, gain, dist)
        cv2.imshow("AprilTag parking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("f"):
            sign *= -1
            print(f"[ctl] drive direction flipped -> {sign:+d}")
        if key == ord(" ") and state == "PARKED":
            state, centred_frames = "SEARCH", 0
            patrol_since = time.monotonic()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="webcam index")
    ap.add_argument("--tag-id", type=int, default=TAG_ID)
    ap.add_argument("--card-serial", default=CARD_SERIAL,
                    help="Connection Card serial (default: %(default)s)")
    ap.add_argument("--no-robot", action="store_true",
                    help="skip Bluetooth; just show the tracking overlay")
    ap.add_argument("--flip-direction", action="store_true",
                    help="start with the drive direction reversed")
    ap.add_argument("--mirror", action="store_true",
                    help="mirror the image (selfie view) before detecting")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"could not open camera {args.camera}")

    car = Car(enabled=not args.no_robot, card_serial=args.card_serial)
    try:
        car.connect()
        park(cap, car, make_detector(), args.tag_id,
             -1 if args.flip_direction else 1, args.mirror)
    except KeyboardInterrupt:
        print()
    finally:
        car.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
