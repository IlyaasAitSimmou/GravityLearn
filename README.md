# KSP Optimal Ascent Autopilot

DEMO Link: https://drive.google.com/file/d/1-ZniaLMHsyDil0PJB-yr0JkCMxqOIzeU/view?usp=sharing

An autopilot that flies a rocket from the launchpad to a circular orbit in
Kerbal Space Program, and does it on as little fuel as it can find. It talks to
the game over [kRPC](https://krpc.github.io/krpc/), and instead of following a
hand-tuned gravity turn it *solves* for one: a physics simulation of the vehicle
runs in the background during the flight, a global optimizer searches that
simulation for the cheapest pitch profile that reaches the target apoapsis, and
the resulting profile is flown live.

Target orbit is set by `TARGET_APOAPSIS` in [basicGravityTurn.py](basicGravityTurn.py)
(default 80 km).

---

## How it works

### 1. Sampling (done before flight)

The optimizer needs a model of the world and a model of the rocket. Two scripts
in [sampleTools/](sampleTools/) build them by querying the game directly, so no
aerodynamic hand-fitting is required.

- [sample_atmosphere.py](sampleTools/sample_atmosphere.py) — walks altitude from
  0 to 70 km in 500 m steps and records density, static pressure, temperature
  and the derived speed of sound. Dumped to `kerbin_atmosphere_table.json`.
  Kerbin's atmosphere never changes, so this is run **once, ever**.
- [sample_vessel_aero.py](sampleTools/sample_vessel_aero.py) — uses kRPC's
  `flight.simulate_aerodynamic_force_at` to build a 3-D drag-area table
  `CdA(altitude, speed, angle of attack)` for the *specific rocket* you are
  flying, written to `vessel_aero_table.json`. It tilts the simulated airflow
  off the vehicle's nose axis to sweep angle of attack, and cross-checks the
  simulated drag against the vehicle's real measured drag to report an accuracy
  ratio. This is run **once per rocket design**, during a short throwaway
  launch after the vehicle is moving (~30+ m/s); the rocket does not need to
  actually fly the whole altitude range.

### 2. The flight

[basicGravityTurn.py](basicGravityTurn.py) sequences three phases:

1. **Vertical ascent** — [phases/vertical_ascent.py](phases/vertical_ascent.py)
   holds 90° pitch until 300 m AGL. Meanwhile the optimizer thread has already
   started, so the trajectory is ready by the time the rocket clears the pad.
2. **Pitch program** — [phases/pitch_program_w_logs.py](phases/pitch_program_w_logs.py)
   is the core of the project (see below).
3. **Circularization** — [phases/circularization.py](phases/circularization.py)
   computes the burn analytically instead of using a maneuver node: Δv from
   vis-viva, burn time from the rocket equation, ignition when time-to-apoapsis
   equals half the burn duration so the burn straddles apoapsis. It returns the
   Δv actually spent, which is added to the ascent Δv for a real mission total.

### 3. The pitch program

The steering law is a cubic in normalized altitude gain:

```
dh    = (altitude - start_altitude) / 40000
pitch = c0 + c1·dh + c2·dh² + c3·dh³
```

`c0` is pinned to the vehicle's actual pitch at hand-over (no discontinuity),
leaving `c1, c2, c3` plus an apoapsis overshoot margin as the four free
parameters. Making pitch a function of **altitude rather than time** made the
program far more robust to modelling error.

Those four numbers are chosen by SciPy's `differential_evolution` over a
trajectory simulator:

- **Integrator** — RK4 on the stacked state `y = (r, v)`, with acceleration
  split into gravity (Newtonian point mass; Kerbin is a perfect sphere in-game,
  so this is exact), thrust (scaled by the sampled ambient-pressure table), and
  drag (trilinear lookup in the sampled `CdA` table).
- **Speed** — every inner-loop routine is `@njit`-compiled with Numba, which is
  what makes running thousands of full trajectory simulations *during* a live
  flight possible. A warm-up thread compiles them while the rocket is still on
  the pad.
- **Cost function** — total fuel needed for ascent *and* circularization, plus a
  quadratic penalty on apoapsis error, a smooth structural penalty on peak
  `q·AoA` load, and a hard penalty if the plan can't afford circularization with
  a 75 m/s Δv reserve.
- **Warm start** — each re-optimization seeds the population with the previous
  solution and one canonical gravity-turn shape, so re-solves converge quickly.

MECO is triggered on predicted apoapsis, then the vehicle coasts to apoapsis for
the circularization burn.

### 4. Logging

The `_w_logs` build writes `flight_log_<timestamp>.csv` into [flight_logs/](flight_logs/)
with 20 columns (altitude, speeds, apoapsis, mass, thrust, drag, CdA, AoA,
commanded vs. actual pitch, Mach, dynamic pressure, density, …). Every row is
tagged `source=pred` (the simulator's own trace of the trajectory it chose) or
`source=act` (live kRPC telemetry at 5 Hz), sharing a common `t=0` at
gravity-turn start — so the entire physics model can be validated against the
game after the flight. [phases/pitch_graphs.py](phases/pitch_graphs.py) plots
these. Use [phases/pitch_program.py](phases/pitch_program.py) for the same
autopilot without the logging overhead.

---

## Running it

**Requirements:** Kerbal Space Program with the kRPC mod installed, and Python
with `krpc`, `numpy`, `scipy`, `numba` (and `matplotlib` for plots). A
virtualenv is included as `KSP_venv/`.

The scripts connect on `rpc_port=50010, stream_port=50011` — set the kRPC server
in-game to match, or edit the `krpc.connect(...)` calls.

```bash
source KSP_venv/bin/activate     # or your own env
```

**Step 1 — atmosphere table (once, ever).** Put any vessel on the pad, then:

```bash
python sampleTools/sample_atmosphere.py
```

**Step 2 — vehicle drag table (once per rocket).** Launch the rocket manually,
let it get airborne and above ~30 m/s, then run:

```bash
python sampleTools/sample_vessel_aero.py
```

Takes a few seconds. Revert the flight afterwards.

**Step 3 — fly.** Put the rocket on the pad, throttle and staging untouched
(the script arms throttle and fires the first stage itself), then:

```bash
python basicGravityTurn.py
```

It will climb vertically, hand over to the optimized pitch program, cut the
engine, coast, circularize, and print the mission Δv breakdown.

> Both JSON tables must exist before step 3, and `vessel_aero_table.json` must
> belong to the rocket you are actually flying — the autopilot's drag model is
> vehicle-specific.

## Notes / limitations

- Single-stage ascent contract: auto-staging is deliberately disabled, because
  `has_fuel` reads `False` for the first few frames after ignition and used to
  jettison the first stage.
- Third-body gravity (Mun, Minmus) is ignored; fine for Kerbin ascent.
- Only the launchpad-to-circular-orbit problem is handled — no rendezvous,
  transfer or landing.

Deeper derivations and code walkthrough live in [code.md](code.md).
