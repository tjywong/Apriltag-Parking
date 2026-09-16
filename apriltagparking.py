"""Stop a LEGO Education car in the middle of the webcam's view.

An AprilTag (tag36h11, id 0) is taped to a tower on the car. The webcam
watches the car drive back and forth parallel to the screen; this script
finds the tag, works out how far its centre sits from the centre of the
frame, and drives the car until that error is ~0 -- then brakes.

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

# --- tuning ----------------------------------------------------------------
TAG_ID = 0                  # the id printed on the tag taped to the car
TAG_DICT = cv2.aruco.DICT_APRILTAG_36H11

KP = 110.0                  # speed (%) per unit of normalised error
MAX_SPEED = 45              # never drive faster than this (%)
MIN_SPEED = 18              # below this the car stalls instead of creeping
PATROL_SPEED = 30           # speed of the back-and-forth search sweep

TOLERANCE = 0.02            # |error| under this (fraction of width) = centred
RELEASE = 0.06              # ...and over this, a parked car starts driving again
HOLD_FRAMES = 4             # consecutive centred frames required to park

PATROL_FLIP_S = 2.5         # reverse the sweep after this long with no tag

LOST_GRACE_S = 0.4          # keep the last command this long before giving up
COMMAND_PERIOD_S = 0.08     # BLE rate limit: at most ~12 commands a second
SPEED_EPSILON = 3           # don't resend a speed that barely changed


class Car:
    """The Double Motor, wrapped so --no-robot can stand in for hardware."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.motor = None
        self._last_speed = None
        self._last_sent = 0.0
        self._last_light = None

    def connect(self):
        if not self.enabled:
            print("[car] --no-robot: running vision only")
            return
        print("[car] scanning for a Double Motor over Bluetooth...")
        self.motor = le.DoubleMotor()
        self.motor.connect()
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


def draw_hud(frame, centre, quad, error, state, speed, sign):
    h, w = frame.shape[:2]
    mid = w // 2
    band = int(TOLERANCE * w)

    cv2.line(frame, (mid, 0), (mid, h), (0, 255, 255), 1)
    cv2.rectangle(frame, (mid - band, 0), (mid + band, h - 1), (0, 200, 200), 1)

    if quad is not None:
        cv2.polylines(frame, [quad.astype(np.int32)], True, (0, 255, 0), 2)
        cx, cy = int(centre[0]), int(centre[1])
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.line(frame, (mid, cy), (cx, cy), (0, 0, 255), 2)

    colour = (0, 255, 0) if state == "PARKED" else (255, 255, 255)
    err = "  --" if error is None else f"{error:+.3f}"
    cv2.putText(frame, f"{state}  err={err}  speed={speed:+d}  dir={sign:+d}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    cv2.putText(frame, "f flip dir   space re-arm   q quit",
                (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


def park(cap, car, detector, tag_id, sign, mirror):
    state = "SEARCH"            # SEARCH -> TRACK -> PARKED
    centred_frames = 0
    patrol_dir = 1
    patrol_since = time.monotonic()
    last_seen = 0.0
    speed = 0

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
            last_seen = now

        if state == "PARKED":
            speed = 0
            if error is not None and abs(error) > RELEASE:
                state, centred_frames = "TRACK", 0

        elif error is not None:
            state = "TRACK"
            if abs(error) <= TOLERANCE:
                centred_frames += 1
                speed = 0
                if centred_frames >= HOLD_FRAMES:
                    state = "PARKED"
                    print(f"[park] centred (err={error:+.3f}) -- braking")
            else:
                centred_frames = 0
                magnitude = min(abs(error) * KP, MAX_SPEED)
                magnitude = max(magnitude, MIN_SPEED)
                # error > 0 means the tag sits right of centre, so drive the
                # way that shrinks it; `sign` absorbs the car's orientation.
                speed = int(round(-np.sign(error) * magnitude * sign))

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

        draw_hud(frame, centre, quad, error, state, speed, sign)
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

    car = Car(enabled=not args.no_robot)
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
