"""Drive a LEGO car up to a stationary AprilTag, seen by a phone on its side.

The phone is mounted on the car facing *sideways*, perpendicular to the
direction of travel, and streams video back to this computer. The car hunts
for a stationary tag36h11 tag, then manoeuvres until the tag is centred in
the phone's view at a set stand-off distance (3 feet by default).

The side mounting is what shapes the manoeuvre. With the tag centred in a
sideways camera it sits exactly abeam, so driving forward changes the range
by nothing at all -- the car just slides past on a tangent. Rather than steer
continuously against that, the car works in measure-move-verify cycles:

    1. stand still and take a fix: range from the tag's apparent size,
       bearing from where it sits in frame
    2. from that fix, work out where the car must stand to be TARGET_MM away
       with the camera facing the tag, and drive there blind
    3. turn the camera back onto the tag and look again, which both checks
       the move and provides the next fix

Every move is therefore open-loop, but every move is also checked, so errors
in the wheel size or the focal length shrink over a few cycles instead of
accumulating. The car only ever measures while stopped, which also sidesteps
motion blur and the phone stream's lag.

    source .venv/bin/activate
    python apriltagwithphone.py --list-cameras      # which index is the phone?
    python apriltagwithphone.py --calibrate 305     # measure the focal length
    python apriltagwithphone.py --focal-px 1400     # then drive, using it
    python apriltagwithphone.py --no-robot          # vision only, no Bluetooth

Defaults assume an iPhone mounted facing the car's RIGHT, arriving over
Continuity Camera as capture index 0. Capture indices do not follow the
order cameras appear in System Settings, so check the preview window.

Keys while the preview window has focus:
    f       flip which side the phone faces (left <-> right)
    y       invert the turn direction
    space   re-arm: leave PARKED and start hunting again
    q/ESC   quit (motors are always stopped on the way out)
"""

import argparse
import statistics
import sys
import threading
import time

import cv2
import numpy as np

import legoeducation as le

# --- hardware --------------------------------------------------------------
CARD_COLOR = le.LEGO_COLOR_ORANGE
CARD_SERIAL = "1129"

TAG_ID = 0
TAG_DICT = cv2.aruco.DICT_APRILTAG_36H11
TAG_MM = 100.0              # printed size of the black border

# --- the goal --------------------------------------------------------------
TARGET_MM = 914.4           # 3 feet, measured camera-to-tag
DIST_TOL_MM = 25.0          # range band that counts as arrived
# How centred is centred. The hub only takes whole-degree turns, so ~1 deg
# is the finest correction available; asking for much less than this just
# makes the car hunt. At 3 feet 1.5 deg is about 35 px off centre in a 1920
# frame, or under 2% of the width.
BEARING_TOL_DEG = 1.5
CENTRE_TRIES = 3            # give up nudging if it stops improving
CENTRE_IMPROVE_DEG = 0.3    # ...where "improving" means this much better

# --- camera ----------------------------------------------------------------
# Range comes from the tag's apparent size, so unlike the fixed-camera version
# this DOES need a focal length. --calibrate measures it; this is the fallback.
# 69 deg is the iPhone 13 Pro Max main (wide) camera: 26 mm equivalent, so
# 2*atan(36/52). Video stabilisation crops into that by roughly a tenth, and
# Center Stage crops dynamically, which would make range meaningless -- turn
# Center Stage off, and prefer --calibrate over this number.
DEFAULT_HFOV_DEG = 69.0

# AprilTag detection costs roughly a pixel's worth of work per pixel. Measured
# on this iPhone stream: 4 detections/s at the native 1920 wide, 16/s at 960,
# 30/s at 640 (camera limited). At 1 foot the tag is still ~230 px across at
# 960, and detection holds to roughly 3 m.
DETECT_WIDTH = 960

# --- drive calibration -----------------------------------------------------
# Turns close on the hub's IMU and straights on wheel odometry, so the only
# physical number we owe the hub is how far one wheel turn carries the car.
# Get it wrong and every move is short or long by that ratio -- which the
# verify step then corrects, so a rough value still converges.
WHEEL_DIAMETER_MM = 56.0    # standard SPIKE wheel; --wheel-mm to override

# --- the manoeuvre ---------------------------------------------------------
# Moving only a fraction of the way each cycle means a wrong wheel size
# shrinks the error instead of overshooting it, and the next look catches
# whatever is left.
APPROACH_GAIN = 0.8
# Once the gap is small the absolute error in a move is small too, so take
# the last one in full rather than creeping in at 80% forever.
FULL_STEP_BELOW_MM = 150.0
MAX_STEP_MM = 700.0         # don't cross the room on one unverified guess
MOVE_SPEED = 35
TURN_SPEED = 30

PROBE_DEG = 25.0            # test turn that reveals which way rotation runs
# The probe is only meaningful if the same tag is in view before and after,
# and if the turn we measure is the only thing that moved. Sanity-check the
# size of the change before believing it.
PROBE_MIN_FRAC = 0.4
PROBE_MAX_FRAC = 2.5
# The hub's IMU measures the probe turn independently of the camera, so the
# same manoeuvre that checks the turn direction also calibrates the lens: if
# a true 25 deg turn slides the tag as though it were 28, the focal length is
# too short. Phone video stabilisation crops the sensor, which does exactly
# that, and an over-read angle inflates the range too.
FOCAL_FIT_MIN = 0.5         # refuse fits outside this band of the estimate
FOCAL_FIT_MAX = 2.0
FOCAL_MIN_TURN_DEG = 8.0    # too small a turn measures mostly noise
SEARCH_STEP_DEG = 25.0      # spin-and-look step while hunting for the tag

OBS_FRAMES = 11             # detections to median together for one fix
OBS_TIMEOUT_S = 1.5         # ...before giving up and calling the tag lost
SETTLE_S = 0.35             # let the camera settle after the car stops

# Telling a wrong `side` from a wrong wheel size, by what the range does:
#   wrong side  -> we aim ~180 deg off and the range grows by about the whole
#                  distance travelled. One move is proof enough.
#   wrong wheel -> the range still moves the right way, just too far or not
#                  far enough, so the next cycle simply trims it.
PARKED_QUIET_MM = 15.0      # once parked, only speak up if the fix shifts
PARKED_QUIET_DEG = 1.0      # ...by more than this

# Hysteresis. Range from a tag's apparent size carries a few cm of noise at
# a metre, so re-engaging at the same threshold we arrived on means noise
# alone sends the car off again. Leave PARKED only on a real change.
RELEASE_MM = 85.0
RELEASE_DEG = 4.0

# A fix that disagrees with the move we just made is not evidence, it is a
# bad measurement -- a partly-seen tag reads small, and small reads far.
# Re-measure rather than act, unless it keeps saying the same thing (the
# car may genuinely have been picked up and moved).
JUMP_SLACK_MM = 150.0
JUMP_RETRIES = 2

RAN_AWAY_FRAC = 0.5         # range grew by this much of the travel = wrong side
BAD_MOVES_BEFORE_FLIP = 2   # weaker fallback for everything else
WORSE_BY_MM = 20.0          # ...where "bad" is range error growing this much

RED = (0, 0, 255)
HUD_BAND = 78


class Stream:
    """Latest-frame reader.

    A network or Continuity stream buffers frames, so a plain read() hands
    back whatever is oldest in the queue -- poison for a control loop. This
    thread keeps only the most recent frame and throws the backlog away.
    """

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:
            pass
        self._lock = threading.Lock()
        self._frame = None
        self._stamp = 0.0
        self._alive = self.cap.isOpened()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        if self._alive:
            self._thread.start()

    def opened(self):
        return self.cap.isOpened()

    def _pump(self):
        misses = 0
        while self._alive:
            ok, frame = self.cap.read()
            if not ok:
                misses += 1
                if misses % 50 == 1:
                    print("[stream] read failed, retrying", file=sys.stderr)
                time.sleep(0.05)
                continue
            misses = 0
            with self._lock:
                self._frame, self._stamp = frame, time.monotonic()

    def latest(self):
        with self._lock:
            if self._frame is None:
                return None, 0.0
            return self._frame.copy(), self._stamp

    def release(self):
        self._alive = False
        self._thread.join(timeout=1.0)
        self.cap.release()


def _color_name(color):
    return le.LEGO_COLOR_NAME_MAP[color].removeprefix("LEGO_COLOR_").lower()


class Car:
    """The Double Motor, driven in discrete measured moves.

    Turns go through the hub's IMU and straights through wheel odometry, so
    these calls finish when the motion is done rather than after a guessed
    interval. `turn_sign` absorbs a car whose idea of left is our right.
    """

    def __init__(self, enabled=True, card_color=CARD_COLOR,
                 card_serial=CARD_SERIAL, wheel_mm=WHEEL_DIAMETER_MM,
                 turn_sign=1):
        self.enabled = enabled
        self.card_color = card_color
        self.card_serial = card_serial
        self.wheel_mm = wheel_mm
        self.turn_sign = turn_sign
        self.motor = None
        self._last_light = None
        # Every BLE command goes through here: the mover thread and the main
        # loop's status light would otherwise write to the hub at once, which
        # is a good way to get "Device disconnected unexpectedly".
        self._lock = threading.RLock()

    def connect(self):
        if not self.enabled:
            print("[car] --no-robot: running vision only")
            return
        print(f"[car] scanning for the Double Motor on Connection Card "
              f"{_color_name(self.card_color)} {self.card_serial}...")
        self.motor = le.DoubleMotor()
        self.motor.connect(card_color=self.card_color,
                           card_serial=self.card_serial)
        # connect() reports failure by printing and returning, not by raising,
        # so without this check we would drive on happily against no hub.
        if not self.motor.connected:
            self.motor = None
            raise ConnectionError(
                f"no Double Motor on Connection Card "
                f"{_color_name(self.card_color)} {self.card_serial}. "
                f"Is the hub powered on, in range, and not still paired to "
                f"another app or an earlier run?")
        self.motor.movement_set_end_state(le.MOTOR_END_STATE_BRAKE)
        print("[car] connected")

    def yaw_deg(self):
        """Hub yaw in degrees, or None before the first IMU notification.

        The hub reports yaw in decidegrees over the range +-1800.
        """
        note = getattr(self.motor, "imu_device", None) if self.motor else None
        return None if note is None else note.yaw / 10.0

    def mm_to_motor_degrees(self, mm):
        return abs(mm) / (np.pi * self.wheel_mm) * 360.0

    def turn_ccw(self, degrees):
        """Turn in place by `degrees`, positive counter-clockwise.

        Returns what the IMU says we actually turned, which is the ground
        truth the camera gets calibrated against.
        """
        degrees *= self.turn_sign
        if abs(degrees) < 1.0:
            return 0.0
        before = self.yaw_deg()
        direction = (le.MOVEMENT_TURN_DIRECTION_LEFT if degrees > 0
                     else le.MOVEMENT_TURN_DIRECTION_RIGHT)
        with self._lock:
            if self.motor is None:
                return 0.0
            self.motor.movement_turn_for_degrees(int(round(abs(degrees))),
                                                 direction=direction,
                                                 speed=TURN_SPEED)
        time.sleep(0.25)            # let the IMU notification catch up
        after = self.yaw_deg()
        if before is None or after is None:
            return 0.0
        return (after - before + 180.0) % 360.0 - 180.0

    def drive_mm(self, mm):
        """Drive straight by `mm`, negative to reverse."""
        if abs(mm) < 5.0:
            return
        direction = (le.MOVEMENT_MOVE_DIRECTION_FORWARD if mm > 0
                     else le.MOVEMENT_MOVE_DIRECTION_BACKWARD)
        with self._lock:
            if self.motor is None:
                return
            self.motor.movement_move_for_degrees(
                int(round(self.mm_to_motor_degrees(mm))),
                direction=direction, speed=MOVE_SPEED)

    def stop(self):
        with self._lock:
            if self.motor is not None:
                self.motor.movement_stop()

    def signal(self, parked):
        if parked == self._last_light:
            return
        self._last_light = parked
        with self._lock:
            if self.motor is not None:
                self.motor.light_color(
                    le.LEGO_COLOR_GREEN if parked else le.LEGO_COLOR_ORANGE,
                    blocking=False)

    def close(self):
        if self.motor is not None:
            self.stop()
            self.motor.disconnect()
            print("[car] disconnected")


def solve_focal(dx1, dx2, turned_deg, focal_guess):
    """Focal length that makes two tag sightings agree with a measured turn.

    A pixel offset dx sits at angle atan(dx/f) from the optical axis, and
    turning the car by a known angle moves the tag by that same angle. One
    unknown, one equation -- bisect for it.
    """
    want = np.radians(abs(turned_deg))

    def spread(f):
        return abs(np.arctan(dx2 / f) - np.arctan(dx1 / f))

    lo, hi = focal_guess * FOCAL_FIT_MIN, focal_guess * FOCAL_FIT_MAX
    # spread() falls as f grows, so the bracket must straddle `want`.
    if not (spread(hi) <= want <= spread(lo)):
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if spread(mid) > want:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def make_detector():
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(TAG_DICT),
                                   params)


def find_tag(detector, frame, tag_id, detect_width=DETECT_WIDTH):
    """Return (centre_xy, corners) for `tag_id`, or (None, None).

    Corners come back in FULL-frame pixels whatever the detection size, so
    range and bearing do not care that detection ran on a smaller image.
    """
    h, w = frame.shape[:2]
    scale = 1.0
    if detect_width and w > detect_width:
        scale = detect_width / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None, None
    for quad, found in zip(corners, ids.flatten()):
        if found == tag_id:
            pts = quad.reshape(4, 2) / scale
            return pts.mean(axis=0), pts
    return None, None


def tag_span(quad):
    """Mean side length of the tag in pixels."""
    sides = [np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)]
    return float(np.mean(sides))


def focal_from_hfov(width_px, hfov_deg=DEFAULT_HFOV_DEG):
    return (width_px / 2.0) / np.tan(np.radians(hfov_deg / 2.0))


def range_mm(span_px, focal_px, offset_px=0.0):
    """Range to the tag from its apparent size.

    TAG_MM * focal / span is the distance along the OPTICAL AXIS, not the
    distance to the tag; the two agree only when the tag is centred. Off to
    one side the real range is longer by 1/cos(bearing), which the hypot
    supplies directly. Skipping this read a tag 30 deg off axis as 84 cm
    when it was really 97 cm -- and during an approach the tag is off axis
    almost by design.
    """
    return TAG_MM * float(np.hypot(focal_px, offset_px)) / max(span_px, 1e-6)


def bearing_deg(cx, width_px, focal_px, side):
    """Tag bearing off the optical axis, positive toward the car's front.

    `side` is +1 when the phone faces the car's left, -1 for its right. A
    left-facing camera has image-right pointing forward; a right-facing one
    has it pointing aft, hence the flip.
    """
    return side * np.degrees(np.arctan2(cx - width_px / 2.0, focal_px))


def plan(dist, bearing, side):
    """Steps to stand TARGET_MM from the tag with the camera facing it.

    The stand-off point lies along the line the camera already looks down, so
    the car only has to travel |dist - TARGET_MM|:

        turn  side*(90 - bearing)   swing the nose onto the tag's direction
        drive dist - TARGET_MM      close (or open) the gap, reversing if we
                                    are already too near
        turn  -side*90              bring the camera back onto the tag

    Those two turns sum to -side*bearing, exactly the rotation that centres
    the tag, so a clean run arrives centred and at range together.
    """
    error = dist - TARGET_MM
    gain = 1.0 if abs(error) < FULL_STEP_BELOW_MM else APPROACH_GAIN
    travel = float(np.clip(error * gain, -MAX_STEP_MM, MAX_STEP_MM))
    return [("turn", side * (90.0 - bearing)),
            ("drive", travel),
            ("turn", -side * 90.0)]


def probe_turn(bearing, side):
    """A test turn, aimed so the tag moves TOWARD centre, not out of frame.

    Turning changes the bearing by side*angle, so this picks the direction
    that shrinks it. Probing the other way once pushed a tag sitting at +20
    deg clean out of a +-34 deg field of view, and the fix taken afterwards
    was no longer comparable.
    """
    towards = -side * (1.0 if bearing >= 0 else -1.0)
    return [("turn", towards * PROBE_DEG)]


def centre_turn(bearing, side):
    """Pure rotation that puts the tag on the optical axis."""
    return [("turn", -side * bearing)]


def describe(steps):
    parts = []
    for kind, amount in steps:
        parts.append(f"turn {amount:+.0f}d" if kind == "turn"
                     else f"drive {amount:+.0f}mm")
    return ", ".join(parts)


def draw_hud(frame, centre, quad, state, dist, bearing, label, side,
             samples, focal_px=None):
    h, w = frame.shape[:2]
    mid = w // 2

    cv2.line(frame, (mid, 0), (mid, h), (0, 255, 255), 1)
    if focal_px:
        band = int(abs(np.tan(np.radians(BEARING_TOL_DEG)) * focal_px))
        cv2.rectangle(frame, (mid - band, 0), (mid + band, h - 1),
                      (0, 200, 200), 1)

    if quad is not None:
        cv2.polylines(frame, [quad.astype(np.int32)], True, RED, 2)
        cx, cy = int(round(centre[0])), int(round(centre[1]))
        cv2.drawMarker(frame, (cx, cy), RED, cv2.MARKER_CROSS, 18, 2)
        cv2.circle(frame, (cx, cy), 4, RED, -1)
        cv2.line(frame, (mid, cy), (cx, cy), RED, 1, cv2.LINE_AA)

        text = f"({cx}, {cy})"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        tx = int(np.clip(cx - tw // 2, 4, w - tw - 4))
        above = int(quad[:, 1].min()) - 10
        ty = above if above - th > HUD_BAND else int(quad[:, 1].max()) + th + 10
        ty = int(np.clip(ty, th + 6, h - base - 4))
        cv2.rectangle(frame, (tx - 5, ty - th - 5), (tx + tw + 5, ty + base + 3),
                      (255, 255, 255), -1)
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    RED, 2, cv2.LINE_AA)

    facing = "left" if side > 0 else "right"
    colour = (0, 255, 0) if state == "PARKED" else (255, 255, 255)
    cv2.putText(frame, f"{state}  phone faces {facing}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    if dist is not None:
        cv2.putText(frame,
                    f"range {dist / 10.0:5.1f} cm (want {TARGET_MM / 10.0:.1f})"
                    f"   bearing {bearing:+5.1f} deg   fix {samples}/{OBS_FRAMES}",
                    (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 120), 2)
    cv2.putText(frame, f"{label}    f flip side   y invert turns   "
                       f"space re-arm   q quit",
                (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


class Mover:
    """Runs a sequence of car steps off the main thread.

    The hub's move calls block until the motion finishes, which would freeze
    the preview, so they run here while the main loop keeps drawing. Only
    this thread touches the motor while a manoeuvre is in flight.
    """

    def __init__(self, car):
        self.car = car
        self.busy = threading.Event()
        self.failed = threading.Event()
        self.label = "idle"
        self.turned = 0.0           # IMU-measured rotation of the last run

    def run(self, steps, label):
        self.label = f"{label}: {describe(steps)}"
        print(f"[move] {self.label}")
        self.busy.set()

        def work():
            turned = 0.0
            try:
                for kind, amount in steps:
                    if kind == "turn":
                        turned += self.car.turn_ccw(amount) or 0.0
                    else:
                        self.car.drive_mm(amount)
                self.turned = turned
            except Exception as exc:
                print(f"[move] failed: {exc}", file=sys.stderr)
                self.failed.set()
            finally:
                self.busy.clear()

        threading.Thread(target=work, daemon=True).start()


def approach(stream, car, detector, tag_id, side, focal_px, mirror,
             detect_width=DETECT_WIDTH, probe=True):
    mover = Mover(car)
    state = "SEARCH"
    label = ""
    last_stamp = 0.0

    samples = []                # fixes collected since the car last stopped
    window_start = None
    resume_at = time.monotonic() + SETTLE_S

    calibrated = not probe
    probe_bearing = None
    probe_expect = 0.0
    probe_offset = 0.0
    centre_tries = 0
    centre_best = None
    parked_at = None
    last_dist = None
    suspect = 0
    pending = None              # what the last move was trying to achieve
    bad_moves = 0
    dist = bearing = None

    while True:
        frame, stamp = stream.latest()
        if frame is None or stamp == last_stamp:
            key = cv2.waitKey(5) & 0xFF
            if key in (ord("q"), 27):
                break
            continue
        last_stamp = stamp
        if mirror:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.monotonic()

        centre, quad = find_tag(detector, frame, tag_id, detect_width)
        fix = None
        if centre is not None:
            offset_px = centre[0] - w / 2.0
            fix = (range_mm(tag_span(quad), focal_px, offset_px),
                   bearing_deg(centre[0], w, focal_px, side),
                   offset_px)
            dist, bearing = fix[0], fix[1]

        if mover.failed.is_set():
            print("[car] lost the hub mid-move -- stopping")
            break

        moving = mover.busy.is_set()
        if moving:
            state = "MOVING"
            samples.clear()
            window_start = None
            resume_at = now + SETTLE_S
        elif now >= resume_at:
            # Stopped and settled: gather a fix, then decide what to do.
            if window_start is None:
                window_start = now
            if fix is not None:
                samples.append(fix)

            ready = len(samples) >= OBS_FRAMES
            timed_out = now - window_start > OBS_TIMEOUT_S
            if ready or timed_out:
                if samples:
                    dist = statistics.median(s[0] for s in samples)
                    bearing = statistics.median(s[1] for s in samples)
                    offset = statistics.median(s[2] for s in samples)
                    hits = len(samples)
                    samples, window_start = [], None

                    # Does this fix square with the move we just made?
                    if last_dist is not None:
                        allowed = abs(pending[1] if pending else 0.0) \
                            + JUMP_SLACK_MM
                        if abs(dist - last_dist) > allowed:
                            suspect += 1
                            if suspect <= JUMP_RETRIES:
                                print(f"[fix] range jumped "
                                      f"{last_dist / 10:.1f} -> "
                                      f"{dist / 10:.1f} cm on a "
                                      f"{abs(pending[1]) if pending else 0:.0f} mm "
                                      f"move -- re-measuring")
                                dist = bearing = None
                                continue
                        else:
                            suspect = 0
                    last_dist = dist

                    # Did the last move do what it promised?
                    if pending is not None:
                        before, travel = pending
                        pending = None
                        grew = dist - before
                        worse = (abs(dist - TARGET_MM)
                                 > abs(before - TARGET_MM) + WORSE_BY_MM)
                        print(f"[fix] range {dist / 10:.1f} cm "
                              f"(was {before / 10:.1f}), bearing "
                              f"{bearing:+.1f} deg, from {hits} frames"
                              f"{'  -- WORSE' if worse else ''}")
                        # Drove away by most of what we travelled? Then we
                        # aimed at the wrong half of the world.
                        ran_away = (abs(travel) > 50.0 and
                                    grew * np.sign(travel)
                                    > RAN_AWAY_FRAC * abs(travel))
                        if ran_away:
                            side *= -1
                            bad_moves = 0
                            print(f"[ctl] moved {abs(travel):.0f} mm and the "
                                  f"range grew {grew:+.0f} mm -- we aimed the "
                                  f"wrong way. The phone faces "
                                  f"{'left' if side > 0 else 'right'}; pass "
                                  f"--side {'left' if side > 0 else 'right'} "
                                  f"to start there.")
                            continue
                        if worse:
                            bad_moves += 1
                            if bad_moves >= BAD_MOVES_BEFORE_FLIP:
                                side *= -1
                                bad_moves = 0
                                print(f"[ctl] moves keep making it worse -- "
                                      f"assuming the phone faces "
                                      f"{'left' if side > 0 else 'right'}.")
                                continue
                        else:
                            bad_moves = 0
                    elif state != "PARKED":
                        print(f"[fix] range {dist / 10:.1f} cm, bearing "
                              f"{bearing:+.1f} deg, from {hits} frames")
                    elif (parked_at is None
                          or abs(dist - parked_at[0]) > PARKED_QUIET_MM
                          or abs(bearing - parked_at[1]) > PARKED_QUIET_DEG):
                        # Parked and still parked: say nothing unless the
                        # world actually moved, or the log is just a wall of
                        # identical lines.
                        print(f"[fix] holding: range {dist / 10:.1f} cm, "
                              f"bearing {bearing:+.1f} deg")
                        parked_at = (dist, bearing)

                    if not calibrated:
                        if probe_bearing is None:
                            steps = probe_turn(bearing, side)
                            probe_bearing = bearing
                            probe_offset = offset
                            probe_expect = side * steps[0][1]
                            state = "PROBE"
                            mover.run(steps, "probe turn")
                        else:
                            moved = bearing - probe_bearing
                            frac = abs(moved) / max(abs(probe_expect), 1e-6)
                            if not PROBE_MIN_FRAC <= frac <= PROBE_MAX_FRAC:
                                # Something other than our turn moved the tag.
                                print(f"[ctl] probe moved the bearing "
                                      f"{moved:+.1f} deg against an expected "
                                      f"{probe_expect:+.1f} -- not believable, "
                                      f"probing again")
                                probe_bearing = None
                            else:
                                # The IMU knows how far we really turned, so
                                # the camera can be calibrated against it.
                                turned = mover.turned
                                if abs(turned) >= FOCAL_MIN_TURN_DEG:
                                    fitted = solve_focal(probe_offset, offset,
                                                         turned, focal_px)
                                    if fitted:
                                        hfov = 2 * np.degrees(
                                            np.arctan((w / 2) / fitted))
                                        print(f"[cal] IMU turned "
                                              f"{turned:+.1f} deg, the tag "
                                              f"moved as if {moved:+.1f} -- "
                                              f"focal {focal_px:.0f} -> "
                                              f"{fitted:.0f} px "
                                              f"(HFOV {hfov:.1f} deg)")
                                        # Every past range was computed with
                                        # the old focal, so rescale rather
                                        # than flag the change as a jump.
                                        if last_dist is not None:
                                            last_dist *= fitted / focal_px
                                        focal_px = fitted
                                    else:
                                        print(f"[cal] focal fit out of range, "
                                              f"keeping {focal_px:.0f} px")
                                if moved * probe_expect < 0:
                                    car.turn_sign *= -1
                                    print(f"[ctl] bearing moved {moved:+.1f} "
                                          f"deg where {probe_expect:+.1f} was "
                                          f"expected -- inverting turn "
                                          f"direction")
                                else:
                                    print(f"[ctl] turn direction confirmed "
                                          f"({moved:+.1f} deg, expected "
                                          f"{probe_expect:+.1f})")
                                calibrated = True
                                probe_bearing = None
                    elif (abs(dist - TARGET_MM)
                          > (RELEASE_MM if state == "PARKED" else DIST_TOL_MM)):
                        state = "MOVE"
                        steps = plan(dist, bearing, side)
                        pending = (dist, steps[1][1])
                        centre_tries, centre_best = 0, None
                        mover.run(steps, "approach")
                    elif (abs(bearing) > (RELEASE_DEG if state == "PARKED"
                                          else BEARING_TOL_DEG)
                          and centre_tries < CENTRE_TRIES):
                        # Only keep nudging while the nudges are helping; a
                        # whole-degree turn cannot resolve much finer.
                        if (centre_best is not None
                                and abs(bearing) > centre_best
                                - CENTRE_IMPROVE_DEG):
                            centre_tries += 1
                        centre_best = min(abs(bearing),
                                          centre_best if centre_best
                                          is not None else abs(bearing))
                        state = "CENTRE"
                        pending = (dist, 0.0)
                        mover.run(centre_turn(bearing, side), "centre")
                    else:
                        if state != "PARKED":
                            off = np.tan(np.radians(bearing)) * focal_px
                            note = ("" if abs(bearing) <= BEARING_TOL_DEG
                                    else "  (as centred as whole-degree turns "
                                         "allow)")
                            print(f"[park] arrived: {dist / 10:.1f} cm, "
                                  f"bearing {bearing:+.1f} deg "
                                  f"= {off:+.0f} px off centre{note}")
                            parked_at = (dist, bearing)
                        state = "PARKED"
                else:
                    samples, window_start = [], None
                    dist = bearing = None
                    state = "SEARCH"
                    pending = None
                    if probe_bearing is not None:
                        # The search spin moves the tag too, so a probe that
                        # straddles a loss measures the wrong thing entirely.
                        print("[ctl] lost the tag mid-probe -- restarting it")
                        probe_bearing = None
                    mover.run([("turn", SEARCH_STEP_DEG)], "search step")

        car.signal(state == "PARKED")
        label = mover.label if state not in ("PARKED",) else "arrived"
        draw_hud(frame, centre, quad, state, dist, bearing, label, side,
                 len(samples), focal_px)
        cv2.imshow("AprilTag approach (phone camera)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("f"):
            side *= -1
            print(f"[ctl] phone now assumed to face "
                  f"{'left' if side > 0 else 'right'}")
        if key == ord("y"):
            car.turn_sign *= -1
            print(f"[ctl] turn direction inverted ({car.turn_sign:+d})")
        if key == ord(" ") and state == "PARKED":
            state, pending, parked_at = "SEARCH", None, None


def calibrate(stream, detector, tag_id, known_mm):
    """Print the focal length implied by a tag held at a known distance."""
    print(f"[cal] hold the tag squarely at {known_mm:.0f} mm; q to finish")
    samples = []
    frame = None
    while True:
        frame, _ = stream.latest()
        if frame is None:
            if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
                break
            continue
        centre, quad = find_tag(detector, frame, tag_id)
        if quad is not None:
            samples.append(tag_span(quad) * known_mm / TAG_MM)
            cv2.polylines(frame, [quad.astype(np.int32)], True, RED, 2)
            median = float(np.median(samples[-90:]))
            cv2.putText(frame, f"focal ~ {median:.0f} px  ({len(samples)} samples)",
                        (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)
        cv2.imshow("calibrate", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    if samples:
        median = float(np.median(samples[-300:]))
        h, w = frame.shape[:2]
        print(f"[cal] focal length ~ {median:.0f} px at {w}x{h}"
              f"  (implied HFOV "
              f"{2 * np.degrees(np.arctan((w / 2) / median)):.0f} deg)")
        print(f"[cal] rerun with:  --focal-px {median:.0f}")
    else:
        print("[cal] no tag seen")


def parse_source(text):
    return int(text) if text.isdigit() else text


def list_cameras(limit=5):
    """Print which capture indices exist.

    Resolution alone will not tell a Continuity Camera apart from a built-in
    one -- both report 1920x1080 here -- and the index order does not match
    the system camera list. Open the preview and look.
    """
    for index in range(limit):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                h, w = frame.shape[:2]
                print(f"  index {index}: {w}x{h}")
            else:
                print(f"  index {index}: opens but no frames")
        cap.release()


def main():
    global TARGET_MM
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0",
                    help="capture index or a stream URL (default: %(default)s, "
                         "the iPhone on this machine). Index order does NOT "
                         "follow the system camera list -- see --list-cameras "
                         "and confirm in the preview window.")
    ap.add_argument("--tag-id", type=int, default=TAG_ID)
    ap.add_argument("--target-mm", type=float, default=TARGET_MM,
                    help="stand-off distance in mm (default: 3 feet)")
    ap.add_argument("--focal-px", type=float, default=None,
                    help="phone focal length in pixels (see --calibrate)")
    ap.add_argument("--hfov", type=float, default=DEFAULT_HFOV_DEG,
                    help="fallback field of view if --focal-px is not given")
    ap.add_argument("--side", choices=("left", "right"), default="left",
                    help="which way the phone faces on the car")
    ap.add_argument("--wheel-mm", type=float, default=WHEEL_DIAMETER_MM,
                    help="driving wheel diameter in mm (default: %(default)s)")
    ap.add_argument("--turn-flip", action="store_true",
                    help="start with the turn direction inverted")
    ap.add_argument("--no-probe", action="store_true",
                    help="skip the test turn that checks rotation direction")
    ap.add_argument("--calibrate", type=float, metavar="MM",
                    help="measure the focal length with the tag at MM")
    ap.add_argument("--card-serial", default=CARD_SERIAL)
    ap.add_argument("--no-robot", action="store_true")
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--detect-width", type=int, default=DETECT_WIDTH,
                    help="downscale to this width before detecting "
                         "(default: %(default)s; 0 disables)")
    ap.add_argument("--list-cameras", action="store_true",
                    help="probe capture indices 0-4 and exit")
    args = ap.parse_args()

    if args.list_cameras:
        list_cameras()
        return

    TARGET_MM = args.target_mm

    stream = Stream(parse_source(args.source))
    if not stream.opened():
        sys.exit(f"could not open stream {args.source!r}")
    detector = make_detector()

    if args.calibrate:
        try:
            calibrate(stream, detector, args.tag_id, args.calibrate)
        finally:
            stream.release()
            cv2.destroyAllWindows()
        return

    deadline = time.monotonic() + 10.0
    frame = None
    while frame is None and time.monotonic() < deadline:
        frame, _ = stream.latest()
    if frame is None:
        stream.release()
        sys.exit("no frames from the stream after 10 s")

    width = frame.shape[1]
    focal_px = args.focal_px or focal_from_hfov(width, args.hfov)
    if args.focal_px is None:
        print(f"[cam] {width}px wide, assuming {args.hfov:.0f} deg HFOV -> "
              f"focal {focal_px:.0f} px. Range is only as good as this; "
              f"use --calibrate for a real number.")

    car = Car(enabled=not args.no_robot, card_serial=args.card_serial,
              wheel_mm=args.wheel_mm, turn_sign=-1 if args.turn_flip else 1)
    try:
        car.connect()
        approach(stream, car, detector, args.tag_id,
                 1 if args.side == "left" else -1, focal_px, args.mirror,
                 args.detect_width, not args.no_probe)
    except KeyboardInterrupt:
        print()
    finally:
        car.close()
        stream.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
