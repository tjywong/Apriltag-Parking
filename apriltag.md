# AprilTag parking — build notes

Three scripts, each a different version of "use an AprilTag to put a LEGO car
where it belongs". They share a tag (tag36h11, id 0, 100 mm) and a hub, and
differ in where the camera sits and how the car is asked to arrive.

| script | camera | job |
|---|---|---|
| `apriltagparking.py` | fixed, watching the car | car drives back and forth, stops centred in frame |
| `apriltagspringparking.py` | fixed, watching the car | same, but overshoots centre and rings down like a spring |
| `apriltagwithphone.py` | phone **on** the car, facing sideways | car finds a stationary tag and parks a set distance from it |

## This rig

Values that are specific to this hardware and were established by measurement,
not assumption:

- **Hub**: Double Motor, Connection Card **orange, serial 1129**.
- **Phone**: iPhone 13 Pro Max over Continuity Camera, capture **index 0**.
  Index order does *not* follow the order cameras appear in System Settings —
  index 1 is the built-in FaceTime camera. Resolution won't tell them apart
  either; both report 1920×1080. Open the preview and look.
- **Phone mounting**: physically faces the car's **right**, but its image
  behaves as **left** (`--side left`, now the default) because of how it sits
  rotated in the mount.
- **Drive direction**: `apriltagspringparking.py` needs `--flip-direction`
  on this build.
- **Focal length**: ~1575–1630 px at 1920 wide, i.e. **62–64° HFOV**, not the
  69° the spec sheet implies. Video stabilisation crops the sensor. Four runs
  agreed on this independently.

Print the tag at 100% scale and check the border with a ruler. Every range
estimate assumes exactly 100 mm.

## `apriltagparking.py` — fixed camera

Tag on the car, camera watching. Error is the tag centroid's horizontal offset
from frame centre, normalised by width.

Two things beyond a plain P controller:

**Distance-scaled gains.** The tag's apparent size is an inverse proxy for
range, so gains, speed cap and stall floor all scale by
`reference_span / measured_span`. Far away the car drives hard; close in it
creeps. It's a ratio of two pixel measurements, so it needs no calibration.

**Lead compensation.** The oscillation around the centre line was *latency*,
not gain. A frame plus the BLE round trip is ~110 ms, and the ±2% deadband is
only ~25 px wide, so a brake command landed after the car had already crossed.
Steering on `error + 0.25 s × error_rate` — where the car *will* be — makes it
brake early.

Measured effect: stopping error went from −0.019/−0.018 (pinned to one edge of
the deadband, the signature of braking on the first qualifying frame) to
−0.001/+0.005/+0.000. Roughly 15× tighter.

## `apriltagspringparking.py` — spring-loaded

Same job, but the car doesn't brake at centre. It sails past, swings back,
overshoots less, and rings down.

The bounce is chosen, not accidental. An unstable controller also oscillates,
but at whatever amplitude its lag happens to produce and as likely to grow as
decay. Here a virtual spring is handed the car's position **and speed** the
moment it reaches the centre band, then evolves

```
x'' = -w^2 x - 2 z w x'
```

with damping ratio `z` from the overshoot asked for and `w` from the period.
The controller chases the spring's position instead of zero. Arriving faster
carries more speed into the spring, so the car swings wider — kinetic energy,
as it should be.

**Counter-intuitive tuning result.** Handing over further out (0.30 instead of
0.12) or asking for a faster swing (1.4 s instead of 2.2 s) both *look* like
they should give a bigger bounce and do the opposite — about a twelfth of the
overshoot. The car saturates, falls behind the spring, and the motion degrades
into lag rather than bounce. The defaults are the measured sweet spot.

`--overshoot` tracks monotonically: 0.10 → 0.08, 0.35 → 0.33, 0.70 → 0.50,
giving swings of 18–114 px at 1920. Springier settings take longer to settle
(10 s at 0.70 versus 4.5 s at 0.35).

## `apriltagwithphone.py` — phone on the car

The side mounting is what shapes this one. **With the tag centred in a
sideways camera it sits exactly abeam, so driving forward changes the range by
nothing at all** — the car slides past on a tangent. Writing `d` for range and
`th` for bearing off the optical axis:

```
d_dot  = -v * sin(th)             range closes only off-axis
th_dot =  w - (v/d) * cos(th)     (v/d)cos th is the rate to ORBIT at d
```

The first design steered continuously, controlling range *through* bearing so
the car spiralled in. It worked in simulation and was miserable in practice —
too many unknown signs and scale constants, and every one of them had to be
right at once.

The current design does **measure-move-verify** cycles instead:

1. stand still and take a fix (median of 11 detections)
2. compute where the car must stand, and drive there blind
3. turn the camera back onto the tag and look again — which both checks the
   move and provides the next fix

The plan per cycle is `turn side*(90 - bearing)`, `drive dist - target`,
`turn -side*90`. Those two turns sum to exactly the rotation that centres the
tag, so a clean run arrives centred and at range together.

Every move is open-loop but every move is checked, so calibration error
shrinks over a few cycles instead of accumulating. Simulation arrives with the
wheel diameter wrong by 0.6× to 1.6×.

**The IMU calibrates the camera.** Turns close on the hub's IMU (`yaw`, in
decidegrees) and straights on wheel odometry, so the hub reports what it
actually did. The probe turn at startup compares the rotation the IMU measured
against how far the tag *appeared* to move, and solves for the focal length
reconciling them. That's how the 69° → ~63° correction above was found. It
matters more the further out the target is: a 10% focal error is ±3 cm at one
foot and ±9 cm at three.

**Off-axis range.** `TAG_MM * focal / span` is the distance along the *optical
axis*, not the distance to the tag; they agree only when the tag is centred.
A pure probe turn — which cannot change the range at all — made the reading
jump 83.7 → 104.2 cm. Multiplying by `1/cos(bearing)` (a `hypot`) narrows that
to 5.7%, the rest being lens distortion at the frame edge. The approach
deliberately holds the tag off-axis, so this was wrong exactly when it counted.

**Self-correcting signs.** A wrong `side` makes the range grow by about the
whole distance travelled, which a wheel-size error never does — so one bad
move is enough to diagnose and flip it.

Best measured run: 151.6 → 93.3 cm in a single move against a 91.4 cm target,
tag 0.9° off centre (24 px at 1920).

## Things that cost time

- **Detection is the bottleneck, not the camera.** The phone streams a clean
  30 fps; AprilTag detection at 1080p was eating 250 ms/frame and capping the
  loop at 4 Hz. Detecting on a downscaled copy and scaling the corners back up
  gives 16 Hz at 960 px wide with no measurable accuracy loss.
- **A tag touching the frame edge is not detected at all.** It needs a white
  quiet zone around the black border, so the tag vanishes from tracking
  slightly *before* it leaves the frame.
- **Turn Center Stage off.** It crops dynamically, which changes the effective
  focal length frame to frame and makes range meaningless.
- **`connect()` doesn't raise on failure** — it prints and returns. Without an
  explicit `motor.connected` check the scripts announced "connected" against
  no hub and then spammed "Attempted to send command without an active
  connection". Both scripts now check.
- **Two threads writing BLE at once will drop the hub.** The main loop's
  status-light call collided with the mover thread mid-command and produced
  "Device disconnected unexpectedly". All motor access goes through a lock.
- **A single dropped camera frame used to end the run.** Continuity Camera
  drops the odd frame and needs a moment to hand over after another process
  releases it; the scripts now retry.
- **Beware self-correcting logic that measures the wrong thing.** Two
  detectors in this project were confidently wrong before they were right: one
  compared a probe turn against a fix taken 300° of searching later, and one
  judged drive direction from `speed × rate` during spring oscillation, where
  phase lag makes that product change sign regardless of the truth. Both now
  only judge in the regime where their assumption holds.

## Running them

```sh
source .venv/bin/activate

# fixed camera
python apriltagparking.py --no-robot          # check tracking first
python apriltagparking.py

# spring-loaded
python apriltagspringparking.py --flip-direction
python apriltagspringparking.py --flip-direction --overshoot 0.6   # looser

# phone on the car
python apriltagwithphone.py --list-cameras    # which index is the phone?
python apriltagwithphone.py                   # defaults suit this rig
python apriltagwithphone.py --target-mm 304.8 # one foot instead of three
```

Keys in the preview window: `q`/ESC quit, `space` re-arm from parked, `f` flip
direction or side, `y` invert turns (phone script).

## Known limits

- ~6% range inconsistency remains across a pure rotation, from lens distortion
  near the frame edge. Fixing it properly means a real camera calibration
  (checkerboard, `cv2.calibrateCamera`) rather than a single focal number.
- The phone script stops as soon as it's inside tolerance, so the final error
  sits near the edge of the band rather than at zero.
- The spring script sometimes loses the tag mid-swing and re-engages; it
  recovers via the search sweep, but a wide swing can carry the tag out of
  frame.
