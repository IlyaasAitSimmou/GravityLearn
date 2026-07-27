"""Drag sweep for the ACTIVE vessel — RUN THIS WHILE FLYING.

Asks kRPC's flight.simulate_aerodynamic_force_at what aerodynamic force THIS
vessel would feel at every (altitude, airspeed, angle-of-attack) grid point
and saves vessel_aero_table.json: the CdA(alt, speed, AoA) lookup table the
ascent sim interpolates.

WORKFLOW (once per rocket): launch vertically at full throttle with SAS on,
and once above ~1 km start this script in a second terminal. The sweep takes
about a minute and never touches the controls; revert to launch afterwards
and fly the real mission with the fresh table.

WHY AIRBORNE: queried from the PAD the game reports 2-4x too much drag
(measured 2026-07-19 with aero_probe.py: the pad-swept table read 4.05x the
real in-flight drag, while the same call made in flight matched reality
within ~5% on average). The vessel needs a live-airflow physics state for
the per-part exposed-area/occlusion data to be right. Pass --pad to force a
pad sweep anyway (comparison runs only).

CdA is normalized with the SAME density table the simulator uses
(kerbin_atmosphere_table.json), so multiplying back by 0.5*rho*v^2 in the sim
reproduces the game's answers at the grid points. Run sample_atmosphere.py
first if that file doesn't exist yet.
"""
import sys
import json
import math
import time

import krpc
import numpy as np

conn = krpc.connect(name="aero sweep", rpc_port=50010, stream_port=50011)
vessel = conn.space_center.active_vessel
body = vessel.orbit.body
# Rotating frame: the atmosphere is static here, so the velocity we pass IS
# the air-relative velocity — no ambiguity about frame rotation.
ref = body.reference_frame
flight = vessel.flight(ref)
radius = body.equatorial_radius

VS = conn.space_center.VesselSituation
if vessel.situation not in (VS.flying, VS.sub_orbital) and "--pad" not in sys.argv:
    print("ERROR: vessel is not flying. Pad sweeps are biased 2-4x high —\n"
          "launch vertically (SAS on, full throttle), get above ~1 km, THEN\n"
          "run this script in a second terminal. The sweep takes ~1 min and\n"
          "doesn't touch the controls; revert to launch when it finishes.\n"
          "(Pass --pad to force a pad sweep anyway, for comparison only.)")
    sys.exit(1)

with open("kerbin_atmosphere_table.json") as f:
    atm = json.load(f)
alt_table = np.array(atm["altitudes"], dtype=np.float64)
rho_table = np.array(atm["densities"], dtype=np.float64)

clamps = vessel.parts.launch_clamps
if clamps:
    print(f"WARNING: {len(clamps)} launch clamp(s) attached — their drag is "
          f"included in this sweep but they detach at liftoff, so the table "
          f"will overestimate drag.")

pos0 = np.array(vessel.position(ref))
up = pos0 / np.linalg.norm(pos0)


def vessel_axes():
    """Current long axis + a perpendicular used to tilt the airflow for AoA
    samples. Re-read every altitude row: the game evaluates forces for the
    vessel's CURRENT orientation, so a stale axis would skew the effective
    AoA as the ship drifts during the sweep. Rockets are axisymmetric, so
    the tilt direction itself is irrelevant."""
    axis = np.array(vessel.direction(ref))
    axis = axis / np.linalg.norm(axis)
    trial = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(axis, trial)) > 0.9:
        trial = np.array([1.0, 0.0, 0.0])
    perp = np.cross(axis, trial)
    return axis, perp / np.linalg.norm(perp)


def self_check():
    """simulate() vs the real drag at the CURRENT flight state — a live
    accuracy metric for this sweep. Should sit near 1.0 while flying; a pad
    sweep reads far higher."""
    vel = np.array(vessel.velocity(ref))
    spd = float(np.linalg.norm(vel))
    if spd < 30.0:
        return None
    f = np.array(flight.simulate_aerodynamic_force_at(
        body, tuple(vessel.position(ref)), tuple(vel)))
    sim_d = max(0.0, float(-np.dot(f, vel / spd)))
    act_d = float(np.linalg.norm(vessel.flight().drag))
    if act_d < 100.0:
        return None
    return sim_d / act_d


# Denser altitude spacing low (where drag matters most), denser speed spacing
# through transonic (where CdA changes fastest). AoA capped at 45 deg — the
# sim clamps lookups beyond the grid edge.
altitudes = [0, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 8000,
             10000, 12000, 14000, 16000, 18000, 21000, 24000, 28000,
             32000, 36000, 40000, 45000, 50000, 55000, 60000, 65000, 69500]
speeds = [10, 30, 60, 100, 140, 180, 220, 250, 270, 290, 310, 330, 350,
          380, 420, 470, 530, 600, 700, 800, 900, 1000, 1150, 1300,
          1500, 1700, 1900, 2100, 2300, 2500]
aoas_deg = [0, 2, 4, 8, 15, 30, 45]

cda = np.zeros((len(altitudes), len(speeds), len(aoas_deg)))
n_total = cda.size
n_done = 0
t_start = time.time()
check_ratios = []

for i, h in enumerate(altitudes):
    axis, perp = vessel_axes()
    ratio = self_check()
    if ratio is not None:
        check_ratios.append(ratio)
    pos = tuple(up * (radius + h))
    rho = float(np.interp(h, alt_table, rho_table))
    for j, v in enumerate(speeds):
        for k, aoa in enumerate(aoas_deg):
            a_rad = math.radians(aoa)
            vel_dir = math.cos(a_rad) * axis + math.sin(a_rad) * perp
            force = np.array(flight.simulate_aerodynamic_force_at(
                body, pos, tuple(vel_dir * float(v))
            ))
            # Drag = component of aero force opposing the airflow direction.
            drag = max(0.0, float(-np.dot(force, vel_dir)))
            if rho > 1e-9:
                cda[i, j, k] = 2.0 * drag / (rho * v * v)
            else:
                cda[i, j, k] = 0.0
    n_done += len(speeds) * len(aoas_deg)
    elapsed = time.time() - t_start
    eta = elapsed / n_done * (n_total - n_done)
    check_str = f", live-check sim/actual={ratio:.2f}" if ratio else ""
    print(f"alt={h:>6}m done  ({n_done}/{n_total}, ETA {eta:.0f}s{check_str})")

mean_check = float(np.mean(check_ratios)) if check_ratios else None

with open("vessel_aero_table.json", "w") as f:
    json.dump({
        "vessel_name": vessel.name,
        "swept_situation": str(vessel.situation),
        "self_check_sim_over_actual": mean_check,
        "altitudes": altitudes,
        "speeds": speeds,
        "aoas_deg": aoas_deg,
        "cda": cda.tolist(),
    }, f)

print(f"\nSaved vessel_aero_table.json for '{vessel.name}' "
      f"({n_total} samples in {time.time() - t_start:.0f}s)")
if mean_check is not None:
    print(f"Live self-check (sim/actual drag at current state): "
          f"{mean_check:.2f} — should be near 1.0")
    if not 0.7 <= mean_check <= 1.4:
        print("WARNING: self-check far from 1.0 — this table is suspect; "
              "re-run while flying faster/higher.")
else:
    print("NOTE: no self-check samples (too slow / pad sweep) — table "
          "accuracy unverified.")

# Sanity check: transonic CdA should be noticeably higher than subsonic.
sub = float(np.interp(100.0, speeds, cda[0, :, 0]))
trans = float(np.interp(370.0, speeds, cda[0, :, 0]))
print(f"AoA 0 at sea level: CdA@100m/s={sub:.3f} m^2 | CdA@370m/s={trans:.3f} m^2 "
      f"(expect a transonic rise, ratio ~2-3x)")
