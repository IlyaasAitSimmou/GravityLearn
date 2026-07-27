"""pitch_program7 + full predicted-vs-actual telemetry CSV logging.

Swap this module in for pitch_program7 in basicGravityTurn.py to record a
flight_log_<timestamp>.csv with one row per sample, tagged `source=pred`
(the sim's step-by-step trace of the trajectory it chose) or `source=act`
(live kRPC telemetry at 5 Hz). Both share the same time origin (t=0 at
gravity-turn start, ~300 m AGL) and the same columns, so the whole physics
model can be compared against the game after the flight.
"""
import time
import json
import csv
import numpy as np
import math
from scipy.optimize import differential_evolution
import threading
from numba import njit

LOG_COLUMNS = [
    "source", "t_s", "phase_burn", "altitude_m", "speed_inertial_ms",
    "h_speed_inertial_ms", "v_speed_ms", "apoapsis_m", "mass_kg", "thrust_N",
    "accel_sensed_ms2", "gravity_ms2", "drag_N", "cda_m2", "aoa_deg",
    "pitch_deg", "pitch_cmd_deg", "mach", "q_pa", "rho_kgm3",
]
N_LOG_COLS = len(LOG_COLUMNS) - 1  # numeric columns (all but `source`)

# ──────────────────────────────────────────────────────────────────────────
# Atmosphere table loader
# ──────────────────────────────────────────────────────────────────────────

def load_atmosphere_table(path="kerbin_atmosphere_table.json"):
    try:
        with open(path) as f:
            table = json.load(f)
    except FileNotFoundError:
        raise RuntimeError(
            f"'{path}' not found. Run sample_atmosphere.py once first "
            f"(any active vessel on the pad is fine) to generate it."
        )
    alt_table = np.array(table["altitudes"], dtype=np.float64)
    rho_table = np.array(table["densities"], dtype=np.float64)
    if "pressures" in table:
        p_table = np.array(table["pressures"], dtype=np.float64)
    else:
        # KSP thrust varies with static pressure, not density. Falling back
        # to density keeps old tables usable but misestimates thrust by the
        # local temperature ratio — regenerate the table to fix.
        print("WARNING: pressures not found in atmosphere table. Run sample_atmosphere.py "
              "again — falling back to density ratio for thrust (less accurate).")
        p_table = rho_table.copy()
    if "speeds_of_sound" in table:
        sos_table = np.array(table["speeds_of_sound"], dtype=np.float64)
    else:
        sos_table = np.ones_like(rho_table) * 300.0
    return alt_table, rho_table, p_table, sos_table


def load_vessel_aero_table(path="vessel_aero_table.json"):
    try:
        with open(path) as f:
            table = json.load(f)
    except FileNotFoundError:
        raise RuntimeError(
            f"'{path}' not found. Run sample_vessel_aero.py once with THIS vessel "
            f"on the pad to sweep its drag profile (takes under a minute)."
        )
    aero_alt_grid = np.array(table["altitudes"], dtype=np.float64)
    aero_speed_grid = np.array(table["speeds"], dtype=np.float64)
    aero_aoa_grid = np.array(table["aoas_deg"], dtype=np.float64)
    cda_table = np.array(table["cda"], dtype=np.float64)
    expected = (len(aero_alt_grid), len(aero_speed_grid), len(aero_aoa_grid))
    if cda_table.shape != expected:
        raise RuntimeError(
            f"vessel aero table shape {cda_table.shape} != grid {expected} — "
            f"re-run sample_vessel_aero.py."
        )
    return aero_alt_grid, aero_speed_grid, aero_aoa_grid, cda_table


# ──────────────────────────────────────────────────────────────────────────
# JIT-Compiled Fast Physics Simulation
# ──────────────────────────────────────────────────────────────────────────

@njit(nogil=True)
def jit_compute_apoapsis(r_vec, v_vec, GM, radius_kerbin):
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    energy = v**2 / 2.0 - GM / r
    if energy >= 0.0:
        return 1e30
    a = -GM / (2.0 * energy)
    h_orb = np.linalg.norm(np.cross(r_vec, v_vec))
    p = h_orb**2 / GM
    e = math.sqrt(abs(1.0 - p / a))
    return (a * (1.0 + e)) - radius_kerbin


@njit(nogil=True)
def jit_atm_lookup(altitude, alt_table, value_table, default_above=0.0):
    if altitude <= alt_table[0]:
        return value_table[0]
    if altitude >= alt_table[-1]:
        return default_above
    idx = np.searchsorted(alt_table, altitude) - 1
    if idx < 0:
        idx = 0
    if idx >= len(alt_table) - 1:
        idx = len(alt_table) - 2
    a0, a1 = alt_table[idx], alt_table[idx + 1]
    v0, v1 = value_table[idx], value_table[idx + 1]
    frac = (altitude - a0) / (a1 - a0)
    return v0 + frac * (v1 - v0)

@njit(nogil=True)
def jit_grid_frac(x, grid):
    n = len(grid)
    if x <= grid[0]:
        return 0, 0.0
    if x >= grid[n - 1]:
        return n - 2, 1.0
    idx = np.searchsorted(grid, x) - 1
    if idx < 0:
        idx = 0
    if idx > n - 2:
        idx = n - 2
    frac = (x - grid[idx]) / (grid[idx + 1] - grid[idx])
    return idx, frac


@njit(nogil=True)
def jit_cda_lookup(altitude, speed, aoa_deg,
                   aero_alt_grid, aero_speed_grid, aero_aoa_grid, cda_table):
    """Trilinear interpolation of the pad-time swept CdA(alt, speed, AoA)
    table (game-sourced via simulate_aerodynamic_force_at). Clamps at the
    grid edges."""
    ia, fa = jit_grid_frac(altitude, aero_alt_grid)
    iv, fv = jit_grid_frac(speed, aero_speed_grid)
    ik, fk = jit_grid_frac(aoa_deg, aero_aoa_grid)

    c00 = cda_table[ia, iv, ik] * (1.0 - fk) + cda_table[ia, iv, ik + 1] * fk
    c01 = cda_table[ia, iv + 1, ik] * (1.0 - fk) + cda_table[ia, iv + 1, ik + 1] * fk
    c10 = cda_table[ia + 1, iv, ik] * (1.0 - fk) + cda_table[ia + 1, iv, ik + 1] * fk
    c11 = cda_table[ia + 1, iv + 1, ik] * (1.0 - fk) + cda_table[ia + 1, iv + 1, ik + 1] * fk

    c0 = c00 * (1.0 - fv) + c01 * fv
    c1 = c10 * (1.0 - fv) + c11 * fv
    return c0 * (1.0 - fa) + c1 * fa


@njit(nogil=True)
def jit_drag_acceleration(r_vec, v_vec, aoa_deg, radius_kerbin,
                          mass, omega_kerbin, alt_table, rho_table,
                          aero_alt_grid, aero_speed_grid, aero_aoa_grid, cda_table):
    future_altitude = np.linalg.norm(r_vec) - radius_kerbin
    if future_altitude > 70000.0:
        return np.zeros(3)

    rho = jit_atm_lookup(future_altitude, alt_table, rho_table, 0.0)
    if rho < 1e-6:
        return np.zeros(3)

    # kRPC body frames are LEFT-handed, so the right-handed np.cross gives the
    # rotation term the opposite sign: physical ground velocity = -np.cross(ω, r).
    omega_vec = np.array([0.0, omega_kerbin, 0.0])
    v_air = v_vec + np.cross(omega_vec, r_vec)
    v_air_speed = np.linalg.norm(v_air)

    if v_air_speed < 0.1:
        return np.zeros(3)

    cda = jit_cda_lookup(future_altitude, v_air_speed, aoa_deg,
                         aero_alt_grid, aero_speed_grid, aero_aoa_grid, cda_table)

    drag_force = 0.5 * rho * v_air_speed**2 * cda
    drag_accel_mag = drag_force / mass
    return -drag_accel_mag * (v_air / v_air_speed)


@njit(nogil=True)
def jit_gravitational_acceleration(r_vec, GM):
    r = np.linalg.norm(r_vec)
    r_unit = r_vec / r
    return -(GM / r**2) * r_unit


@njit(nogil=True)
def jit_accel(t, r_vec, v_vec, mass, thrust_magnitude,
              radius_kerbin,
              GM, omega_kerbin,
              start_alt, c0, c1, c2, c3,
              thrust_enabled, alt_table, rho_table,
              aero_alt_grid, aero_speed_grid, aero_aoa_grid, cda_table):
    """pitch(dh) = c0 + c1*dh + c2*dh² + c3*dh³"""
    altitude = np.linalg.norm(r_vec) - radius_kerbin
    dh = (altitude - start_alt) / 40000.0
    if dh < 0.0: dh = 0.0
    pitch = c0 + c1 * dh + c2 * dh**2 + c3 * dh**3
    if pitch < 0.0:
        pitch = 0.0
    if pitch > 90.0:
        pitch = 90.0
    pitch_rad = math.radians(pitch)

    r_unit = r_vec / np.linalg.norm(r_vec)
    north = np.array([0.0, 1.0, 0.0])
    # Left-handed frame: np.cross(north, r_unit) points true WEST — swap args.
    east = np.cross(r_unit, north)
    east = east / np.linalg.norm(east)

    thrust_direction = math.cos(pitch_rad) * east + math.sin(pitch_rad) * r_unit

    if thrust_enabled:
        a_thrust = (thrust_magnitude / mass) * thrust_direction
    else:
        a_thrust = np.zeros(3)

    v_air = v_vec + np.cross(np.array([0.0, omega_kerbin, 0.0]), r_vec)
    v_air_speed = np.linalg.norm(v_air)
    # Post-MECO the vehicle coasts surface-prograde (run_program commands it),
    # so AoA is 0 — the cubic's tail is unfitted extrapolation, not attitude.
    if thrust_enabled and v_air_speed > 1.0:
        cos_aoa = np.dot(thrust_direction, v_air / v_air_speed)
        cos_aoa = max(-1.0, min(1.0, cos_aoa))
        aoa_deg = math.degrees(math.acos(cos_aoa))
    else:
        aoa_deg = 0.0

    a_gravity = jit_gravitational_acceleration(r_vec, GM)
    a_drag = jit_drag_acceleration(r_vec, v_vec, aoa_deg,
                                   radius_kerbin, mass, omega_kerbin,
                                   alt_table, rho_table,
                                   aero_alt_grid, aero_speed_grid, aero_aoa_grid, cda_table)
    return a_thrust + a_gravity + a_drag




@njit(nogil=True)
def jit_sim_with_coast(r_start, v_start, fuel_mass, m_start, dry_mass,
                       mass_flow_rate, max_sl_thrust, max_vac_thrust,
                       radius_kerbin, GM,
                       omega_kerbin,
                       start_alt, c0, c1, c2, c3, apo_meco_trigger,
                       dt_step, max_time, alt_table, rho_table, p_table,
                       aero_alt_grid, aero_speed_grid, aero_aoa_grid, cda_table):
    """Powered ascent → MECO (apoapsis trigger or fuel exhaustion) →
    atmospheric coast until clear of atmosphere or apoapsis reached.
    """
    dt = dt_step
    r = r_start.copy()
    v = v_start.copy()
    m = m_start
    fuel = fuel_mass
    t = 0.0
    thrust_enabled = True
    meco_occurred = False
    meco_time = -1.0
    meco_alt = -1.0
    max_load = 0.0

    accumulated_dv = 0.0

    while t < max_time:
        if thrust_enabled:
            inst_apo = jit_compute_apoapsis(r, v, GM, radius_kerbin)
            if inst_apo >= apo_meco_trigger:
                thrust_enabled = False
                meco_occurred = True
                meco_time = t
                meco_alt = np.linalg.norm(r) - radius_kerbin
                # MECO precision no longer needed — restore the coarse clock
                # so the (possibly long) coast doesn't run at 10x the steps.
                dt = dt_step
            elif apo_meco_trigger - inst_apo < 8000.0:
                # FIX Q1: Slow down the simulation clock when approaching MECO
                # to prevent overshooting the target in a single massive tick.
                dt = 0.1

        omega_vec = np.array([0.0, omega_kerbin, 0.0])
        v_air = v + np.cross(omega_vec, r)
        v_air_speed = np.linalg.norm(v_air)
        altitude = np.linalg.norm(r) - radius_kerbin
        
        r_unit = r / np.linalg.norm(r)
        east = np.cross(r_unit, np.array([0.0, 1.0, 0.0]))
        if np.linalg.norm(east) < 1e-6:
            east = np.array([1.0, 0.0, 0.0])
        else:
            east = east / np.linalg.norm(east)
            
        alt_current = np.linalg.norm(r) - radius_kerbin
        dh = (alt_current - start_alt) / 40000.0
        if dh < 0.0: dh = 0.0
        pitch_deg = c0 + c1 * dh + c2 * dh**2 + c3 * dh**3
        pitch_deg = max(0.0, min(90.0, pitch_deg))
        pitch_rad = math.radians(pitch_deg)
        thrust_direction = r_unit * math.sin(pitch_rad) + east * math.cos(pitch_rad)
        
        # Coast is flown surface-prograde → AoA 0 (also keeps the max_load
        # tracker burn-phase-only instead of reading the unfitted cubic tail).
        if thrust_enabled and v_air_speed > 1.0:
            cos_aoa = np.dot(thrust_direction, v_air / v_air_speed)
            cos_aoa = max(-1.0, min(1.0, cos_aoa))
            aoa_deg = math.degrees(math.acos(cos_aoa))
        else:
            aoa_deg = 0.0
            
        rho = np.interp(altitude, alt_table, rho_table)
        q = 0.5 * rho * v_air_speed**2
        load = q * aoa_deg
        if load > max_load:
            max_load = load

        alt1 = np.linalg.norm(r) - radius_kerbin
        if thrust_enabled:
            th1_val = 1.0
            dm1 = mass_flow_rate * th1_val
        else:
            th1_val = 0.0
            dm1 = 0.0
            
        p1 = np.interp(alt1, alt_table, p_table)
        pressure_ratio1 = min(1.0, max(0.0, p1 / p_table[0]))
        thrust_magnitude1 = max_vac_thrust + (max_sl_thrust - max_vac_thrust) * pressure_ratio1

        k1_v = jit_accel(t, r, v, m, thrust_magnitude1 * th1_val,
                         radius_kerbin,
                         GM, omega_kerbin, start_alt, c0, c1, c2, c3,
                         thrust_enabled, alt_table, rho_table,
                         aero_alt_grid, aero_speed_grid, aero_aoa_grid, cda_table)
        k1_r = v

        r2 = r + k1_r * dt / 2.0
        alt2 = np.linalg.norm(r2) - radius_kerbin
        if thrust_enabled:
            th2_val = 1.0
            dm2 = mass_flow_rate * th2_val
        else:
            th2_val = 0.0
            dm2 = 0.0
            
        m2 = max(dry_mass, m - dm1 * dt / 2.0)
        v2 = v + k1_v * dt / 2.0
        p2 = np.interp(alt2, alt_table, p_table)
        pressure_ratio2 = min(1.0, max(0.0, p2 / p_table[0]))
        thrust_magnitude2 = max_vac_thrust + (max_sl_thrust - max_vac_thrust) * pressure_ratio2
        k2_v = jit_accel(t + dt / 2.0, r2, v2, m2, thrust_magnitude2 * th2_val,
                         radius_kerbin,
                         GM, omega_kerbin, start_alt, c0, c1, c2, c3,
                         thrust_enabled, alt_table, rho_table,
                         aero_alt_grid, aero_speed_grid, aero_aoa_grid, cda_table)
        k2_r = v2

        r3 = r + k2_r * dt / 2.0
        alt3 = np.linalg.norm(r3) - radius_kerbin
        if thrust_enabled:
            th3_val = 1.0
            dm3 = mass_flow_rate * th3_val
        else:
            th3_val = 0.0
            dm3 = 0.0
            
        m3 = max(dry_mass, m - dm2 * dt / 2.0)
        v3 = v + k2_v * dt / 2.0
        p3 = np.interp(alt3, alt_table, p_table)
        pressure_ratio3 = min(1.0, max(0.0, p3 / p_table[0]))
        thrust_magnitude3 = max_vac_thrust + (max_sl_thrust - max_vac_thrust) * pressure_ratio3
        k3_v = jit_accel(t + dt / 2.0, r3, v3, m3, thrust_magnitude3 * th3_val,
                         radius_kerbin,
                         GM, omega_kerbin, start_alt, c0, c1, c2, c3,
                         thrust_enabled, alt_table, rho_table,
                         aero_alt_grid, aero_speed_grid, aero_aoa_grid, cda_table)
        k3_r = v3

        r4 = r + k3_r * dt
        alt4 = np.linalg.norm(r4) - radius_kerbin
        if thrust_enabled:
            th4_val = 1.0
            dm4 = mass_flow_rate * th4_val
        else:
            th4_val = 0.0
            dm4 = 0.0
            
        m4 = max(dry_mass, m - dm3 * dt)
        v4 = v + k3_v * dt
        p4 = np.interp(alt4, alt_table, p_table)
        pressure_ratio4 = min(1.0, max(0.0, p4 / p_table[0]))
        thrust_magnitude4 = max_vac_thrust + (max_sl_thrust - max_vac_thrust) * pressure_ratio4
        k4_v = jit_accel(t + dt, r4, v4, m4, thrust_magnitude4 * th4_val,
                         radius_kerbin,
                         GM, omega_kerbin, start_alt, c0, c1, c2, c3,
                         thrust_enabled, alt_table, rho_table,
                         aero_alt_grid, aero_speed_grid, aero_aoa_grid, cda_table)
        k4_r = v4

        v = v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        r = r + (dt / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)

        if thrust_enabled:
            avg_dm = (dm1 + 2.0 * dm2 + 2.0 * dm3 + dm4) / 6.0
            fuel -= avg_dm * dt

            # FIX Q3: Calculate the average thrust over this RK4 step and integrate the dv
            avg_thrust = (thrust_magnitude1 + 2.0*thrust_magnitude2 + 2.0*thrust_magnitude3 + thrust_magnitude4) / 6.0
            accumulated_dv += (avg_thrust / m) * dt

            if fuel <= 0.0:
                fuel = 0.0
                thrust_enabled = False
                meco_occurred = True
                meco_time = t
                meco_alt = np.linalg.norm(r) - radius_kerbin
                dt = dt_step

        m = dry_mass + fuel
        t += dt

        altitude = np.linalg.norm(r) - radius_kerbin
        r_unit = r / np.linalg.norm(r)
        vertical_velocity = np.dot(v, r_unit)

        if meco_occurred:
            # SUCCESS: Above atmosphere, coast is complete
            if altitude > 70000.0:
                break
            # FAIL: Rocket fell back below starting altitude (trajectory crashed)
            if altitude < start_alt:
                break

        if altitude < 0.0:
            break
        if not meco_occurred and vertical_velocity < -50.0:
            break

    return r, v, m, meco_occurred, meco_time, max_load, meco_alt, accumulated_dv


@njit(nogil=True)
def jit_sim_with_coast_logged(r_start, v_start, fuel_mass, m_start, dry_mass,
                              mass_flow_rate, max_sl_thrust, max_vac_thrust,
                              radius_kerbin, GM,
                              omega_kerbin,
                              start_alt, c0, c1, c2, c3, apo_meco_trigger,
                              dt_step, max_time, alt_table, rho_table, p_table,
                              aero_alt_grid, aero_speed_grid, aero_aoa_grid, cda_table,
                              sos_table, log_out):
    """EXACT physics copy of jit_sim_with_coast, plus one telemetry row per
    integrator step written into log_out (columns = LOG_COLUMNS minus
    `source`, state sampled at the START of each step). Runs ONCE per flight
    to produce the predicted trace, so the per-step bookkeeping cost is
    irrelevant. KEEP THE PHYSICS IN SYNC with jit_sim_with_coast."""
    dt = dt_step
    r = r_start.copy()
    v = v_start.copy()
    m = m_start
    fuel = fuel_mass
    t = 0.0
    thrust_enabled = True
    meco_occurred = False
    n_rows = 0
    max_rows = log_out.shape[0]

    while t < max_time:
        if thrust_enabled:
            inst_apo = jit_compute_apoapsis(r, v, GM, radius_kerbin)
            if inst_apo >= apo_meco_trigger:
                thrust_enabled = False
                meco_occurred = True
                dt = dt_step
            elif apo_meco_trigger - inst_apo < 8000.0:
                dt = 0.1

        omega_vec = np.array([0.0, omega_kerbin, 0.0])
        v_air = v + np.cross(omega_vec, r)
        v_air_speed = np.linalg.norm(v_air)
        altitude = np.linalg.norm(r) - radius_kerbin

        r_unit = r / np.linalg.norm(r)
        east = np.cross(r_unit, np.array([0.0, 1.0, 0.0]))
        if np.linalg.norm(east) < 1e-6:
            east = np.array([1.0, 0.0, 0.0])
        else:
            east = east / np.linalg.norm(east)

        dh = (altitude - start_alt) / 40000.0
        if dh < 0.0: dh = 0.0
        pitch_deg = c0 + c1 * dh + c2 * dh**2 + c3 * dh**3
        pitch_deg = max(0.0, min(90.0, pitch_deg))
        pitch_rad = math.radians(pitch_deg)
        thrust_direction = r_unit * math.sin(pitch_rad) + east * math.cos(pitch_rad)

        # Coast is flown surface-prograde → AoA 0 (also keeps the max_load
        # tracker burn-phase-only instead of reading the unfitted cubic tail).
        if thrust_enabled and v_air_speed > 1.0:
            cos_aoa = np.dot(thrust_direction, v_air / v_air_speed)
            cos_aoa = max(-1.0, min(1.0, cos_aoa))
            aoa_deg = math.degrees(math.acos(cos_aoa))
        else:
            aoa_deg = 0.0

        rho = np.interp(altitude, alt_table, rho_table)
        q = 0.5 * rho * v_air_speed**2

        # ── telemetry row ────────────────────────────────────────────
        if n_rows < max_rows:
            p_here = np.interp(altitude, alt_table, p_table)
            pressure_ratio = min(1.0, max(0.0, p_here / p_table[0]))
            if thrust_enabled:
                thrust_now = max_vac_thrust + (max_sl_thrust - max_vac_thrust) * pressure_ratio
            else:
                thrust_now = 0.0
            a_drag_vec = jit_drag_acceleration(r, v, aoa_deg, radius_kerbin, m,
                                               omega_kerbin, alt_table, rho_table,
                                               aero_alt_grid, aero_speed_grid,
                                               aero_aoa_grid, cda_table)
            a_thrust_vec = (thrust_now / m) * thrust_direction
            a_sensed = np.linalg.norm(a_thrust_vec + a_drag_vec)
            drag_N = np.linalg.norm(a_drag_vec) * m
            cda_here = jit_cda_lookup(altitude, v_air_speed, aoa_deg,
                                      aero_alt_grid, aero_speed_grid,
                                      aero_aoa_grid, cda_table)
            speed = np.linalg.norm(v)
            vr = np.dot(v, r_unit)
            h_sq = speed * speed - vr * vr
            h_speed = math.sqrt(h_sq) if h_sq > 0.0 else 0.0
            apo_now = jit_compute_apoapsis(r, v, GM, radius_kerbin)
            sos = jit_atm_lookup(altitude, alt_table, sos_table, 300.0)
            mach = v_air_speed / sos if sos > 1.0 else 0.0

            log_out[n_rows, 0] = t
            log_out[n_rows, 1] = 1.0 if thrust_enabled else 0.0
            log_out[n_rows, 2] = altitude
            log_out[n_rows, 3] = speed
            log_out[n_rows, 4] = h_speed
            log_out[n_rows, 5] = vr
            log_out[n_rows, 6] = apo_now
            log_out[n_rows, 7] = m
            log_out[n_rows, 8] = thrust_now
            log_out[n_rows, 9] = a_sensed
            log_out[n_rows, 10] = GM / (np.linalg.norm(r) ** 2)
            log_out[n_rows, 11] = drag_N
            log_out[n_rows, 12] = cda_here
            log_out[n_rows, 13] = aoa_deg
            log_out[n_rows, 14] = pitch_deg
            log_out[n_rows, 15] = pitch_deg
            log_out[n_rows, 16] = mach
            log_out[n_rows, 17] = q
            log_out[n_rows, 18] = rho
            n_rows += 1

        # ── RK4 step (identical to jit_sim_with_coast) ───────────────
        alt1 = np.linalg.norm(r) - radius_kerbin
        if thrust_enabled:
            th1_val = 1.0
            dm1 = mass_flow_rate * th1_val
        else:
            th1_val = 0.0
            dm1 = 0.0

        p1 = np.interp(alt1, alt_table, p_table)
        pressure_ratio1 = min(1.0, max(0.0, p1 / p_table[0]))
        thrust_magnitude1 = max_vac_thrust + (max_sl_thrust - max_vac_thrust) * pressure_ratio1

        k1_v = jit_accel(t, r, v, m, thrust_magnitude1 * th1_val,
                         radius_kerbin,
                         GM, omega_kerbin, start_alt, c0, c1, c2, c3,
                         thrust_enabled, alt_table, rho_table,
                         aero_alt_grid, aero_speed_grid, aero_aoa_grid, cda_table)
        k1_r = v

        r2 = r + k1_r * dt / 2.0
        alt2 = np.linalg.norm(r2) - radius_kerbin
        if thrust_enabled:
            th2_val = 1.0
            dm2 = mass_flow_rate * th2_val
        else:
            th2_val = 0.0
            dm2 = 0.0

        m2 = max(dry_mass, m - dm1 * dt / 2.0)
        v2 = v + k1_v * dt / 2.0
        p2 = np.interp(alt2, alt_table, p_table)
        pressure_ratio2 = min(1.0, max(0.0, p2 / p_table[0]))
        thrust_magnitude2 = max_vac_thrust + (max_sl_thrust - max_vac_thrust) * pressure_ratio2
        k2_v = jit_accel(t + dt / 2.0, r2, v2, m2, thrust_magnitude2 * th2_val,
                         radius_kerbin,
                         GM, omega_kerbin, start_alt, c0, c1, c2, c3,
                         thrust_enabled, alt_table, rho_table,
                         aero_alt_grid, aero_speed_grid, aero_aoa_grid, cda_table)
        k2_r = v2

        r3 = r + k2_r * dt / 2.0
        alt3 = np.linalg.norm(r3) - radius_kerbin
        if thrust_enabled:
            th3_val = 1.0
            dm3 = mass_flow_rate * th3_val
        else:
            th3_val = 0.0
            dm3 = 0.0

        m3 = max(dry_mass, m - dm2 * dt / 2.0)
        v3 = v + k2_v * dt / 2.0
        p3 = np.interp(alt3, alt_table, p_table)
        pressure_ratio3 = min(1.0, max(0.0, p3 / p_table[0]))
        thrust_magnitude3 = max_vac_thrust + (max_sl_thrust - max_vac_thrust) * pressure_ratio3
        k3_v = jit_accel(t + dt / 2.0, r3, v3, m3, thrust_magnitude3 * th3_val,
                         radius_kerbin,
                         GM, omega_kerbin, start_alt, c0, c1, c2, c3,
                         thrust_enabled, alt_table, rho_table,
                         aero_alt_grid, aero_speed_grid, aero_aoa_grid, cda_table)
        k3_r = v3

        r4 = r + k3_r * dt
        alt4 = np.linalg.norm(r4) - radius_kerbin
        if thrust_enabled:
            th4_val = 1.0
            dm4 = mass_flow_rate * th4_val
        else:
            th4_val = 0.0
            dm4 = 0.0

        m4 = max(dry_mass, m - dm3 * dt)
        v4 = v + k3_v * dt
        p4 = np.interp(alt4, alt_table, p_table)
        pressure_ratio4 = min(1.0, max(0.0, p4 / p_table[0]))
        thrust_magnitude4 = max_vac_thrust + (max_sl_thrust - max_vac_thrust) * pressure_ratio4
        k4_v = jit_accel(t + dt, r4, v4, m4, thrust_magnitude4 * th4_val,
                         radius_kerbin,
                         GM, omega_kerbin, start_alt, c0, c1, c2, c3,
                         thrust_enabled, alt_table, rho_table,
                         aero_alt_grid, aero_speed_grid, aero_aoa_grid, cda_table)
        k4_r = v4

        v = v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        r = r + (dt / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)

        if thrust_enabled:
            avg_dm = (dm1 + 2.0 * dm2 + 2.0 * dm3 + dm4) / 6.0
            fuel -= avg_dm * dt
            if fuel <= 0.0:
                fuel = 0.0
                thrust_enabled = False
                meco_occurred = True
                dt = dt_step

        m = dry_mass + fuel
        t += dt

        altitude = np.linalg.norm(r) - radius_kerbin
        r_unit = r / np.linalg.norm(r)
        vertical_velocity = np.dot(v, r_unit)

        if meco_occurred:
            if altitude > 70000.0:
                break
            if altitude < start_alt:
                break

        if altitude < 0.0:
            break
        if not meco_occurred and vertical_velocity < -50.0:
            break

    return r, v, m, meco_occurred, n_rows


@njit(nogil=True)
def jit_sim_vertical_climb(
    r_start,
    v_start,
    m_start,
    fuel_mass,
    mass_flow_rate,
    max_sl_thrust,
    max_vac_thrust,
    radius_kerbin,
    GM,
    omega_kerbin,
    target_altitude,
    dt,
    max_time,
    alt_table,
    rho_table,
    p_table,
    aero_alt_grid,
    aero_speed_grid,
    aero_aoa_grid,
    cda_table
):
    r = r_start.copy()
    v = v_start.copy()

    fuel = fuel_mass
    dry_mass = m_start - fuel_mass
    m = m_start

    t = 0.0
    accumulated_dv = 0.0

    while t < max_time:

        altitude = np.linalg.norm(r) - radius_kerbin

        if altitude >= target_altitude:
            break

        if fuel <= 0.0:
            break

        r_unit = r / np.linalg.norm(r)

        p = np.interp(altitude, alt_table, p_table)
        pressure_ratio = min(1.0, max(0.0, p / p_table[0]))
        thrust_magnitude = max_vac_thrust + (max_sl_thrust - max_vac_thrust) * pressure_ratio

        a_thrust = (thrust_magnitude / m) * r_unit
        a_gravity = -(GM / np.dot(r, r)) * r_unit
        a_drag = jit_drag_acceleration(r, v, 0.0, radius_kerbin, m, omega_kerbin, alt_table, rho_table,
                                       aero_alt_grid, aero_speed_grid, aero_aoa_grid, cda_table)

        a = a_thrust + a_gravity + a_drag

        k1_v = a
        k1_r = v

        m2 = max(dry_mass, m - mass_flow_rate * dt * 0.5)
        r2 = r + k1_r * dt * 0.5
        v2 = v + k1_v * dt * 0.5
        r2_unit = r2 / np.linalg.norm(r2)
        p2 = np.interp(np.linalg.norm(r2) - radius_kerbin, alt_table, p_table)
        thrust_mag2 = max_vac_thrust + (max_sl_thrust - max_vac_thrust) * min(1.0, max(0.0, p2 / p_table[0]))
        a2_thrust = (thrust_mag2 / m2) * r2_unit
        a2_gravity = -(GM / np.dot(r2, r2)) * r2_unit
        a2_drag = jit_drag_acceleration(r2, v2, 0.0, radius_kerbin, m2, omega_kerbin, alt_table, rho_table,
                                        aero_alt_grid, aero_speed_grid, aero_aoa_grid, cda_table)

        k2_v = a2_thrust + a2_gravity + a2_drag
        k2_r = v2

        m3 = max(dry_mass, m - mass_flow_rate * dt * 0.5)
        r3 = r + k2_r * dt * 0.5
        v3 = v + k2_v * dt * 0.5
        r3_unit = r3 / np.linalg.norm(r3)
        p3 = np.interp(np.linalg.norm(r3) - radius_kerbin, alt_table, p_table)
        thrust_mag3 = max_vac_thrust + (max_sl_thrust - max_vac_thrust) * min(1.0, max(0.0, p3 / p_table[0]))
        a3_thrust = (thrust_mag3 / m3) * r3_unit
        a3_gravity = -(GM / np.dot(r3, r3)) * r3_unit
        a3_drag = jit_drag_acceleration(r3, v3, 0.0, radius_kerbin, m3, omega_kerbin, alt_table, rho_table,
                                        aero_alt_grid, aero_speed_grid, aero_aoa_grid, cda_table)

        k3_v = a3_thrust + a3_gravity + a3_drag
        k3_r = v3

        m4 = max(dry_mass, m - mass_flow_rate * dt)
        r4 = r + k3_r * dt
        v4 = v + k3_v * dt
        r4_unit = r4 / np.linalg.norm(r4)
        p4 = np.interp(np.linalg.norm(r4) - radius_kerbin, alt_table, p_table)
        thrust_mag4 = max_vac_thrust + (max_sl_thrust - max_vac_thrust) * min(1.0, max(0.0, p4 / p_table[0]))
        a4_thrust = (thrust_mag4 / m4) * r4_unit
        a4_gravity = -(GM / np.dot(r4, r4)) * r4_unit
        a4_drag = jit_drag_acceleration(r4, v4, 0.0, radius_kerbin, m4, omega_kerbin, alt_table, rho_table,
                                        aero_alt_grid, aero_speed_grid, aero_aoa_grid, cda_table)

        k4_v = a4_thrust + a4_gravity + a4_drag
        k4_r = v4

        v = v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        r = r + (dt / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)

        accumulated_dv += (thrust_magnitude / m) * dt
        fuel -= mass_flow_rate * dt
        m = dry_mass + fuel
        t += dt

    return r, v, m, t, accumulated_dv


# ──────────────────────────────────────────────────────────────────────────
# Main Class
# ──────────────────────────────────────────────────────────────────────────

class PitchProgram():
    def __init__(self, vessel, conn):
        self.vessel = vessel
        self.conn = conn

        kerbin = vessel.orbit.body
        self.ref = kerbin.non_rotating_reference_frame

        self.surface_altitude = conn.add_stream(getattr, vessel.flight(), 'surface_altitude')
        self.apoapsis = conn.add_stream(getattr, vessel.orbit, 'apoapsis_altitude')
        self.thrust = conn.add_stream(getattr, vessel, 'thrust')
        self.isp = conn.add_stream(getattr, vessel, 'specific_impulse')
        self.mass = conn.add_stream(getattr, vessel, 'mass')
        self.direction = conn.add_stream(vessel.direction, self.ref)
        self.rho = conn.add_stream(getattr, vessel.flight(), 'atmosphere_density')
        self.g_force = conn.add_stream(getattr, vessel.flight(), 'g_force')

        self.GM = vessel.orbit.body.gravitational_parameter
        self.radius_kerbin = vessel.orbit.body.equatorial_radius
        self.omega_kerbin = vessel.orbit.body.rotational_speed

        self.position = conn.add_stream(vessel.position, self.ref)
        self.velocity = conn.add_stream(vessel.velocity, self.ref)

        # FIX: load real density-vs-altitude table from file instead of
        # live-sampling — instant startup, samples KSP's true curve once.
        (self.alt_table, self.rho_table,
         self.p_table, self.sos_table) = load_atmosphere_table()

        # ── Telemetry streams + state for the flight logger ─────────
        flight_inertial = vessel.flight(self.ref)
        flight_srf = vessel.flight()
        self.ut = conn.add_stream(getattr, conn.space_center, 'ut')
        self._s_mean_alt = conn.add_stream(getattr, flight_inertial, 'mean_altitude')
        self._s_speed = conn.add_stream(getattr, flight_inertial, 'speed')
        self._s_hspeed = conn.add_stream(getattr, flight_inertial, 'horizontal_speed')
        self._s_vspeed = conn.add_stream(getattr, flight_inertial, 'vertical_speed')
        self._s_drag = conn.add_stream(getattr, flight_srf, 'drag')
        self._s_tas = conn.add_stream(getattr, flight_srf, 'true_air_speed')
        self._s_aoa = conn.add_stream(getattr, flight_srf, 'angle_of_attack')
        self._s_pitch = conn.add_stream(getattr, flight_srf, 'pitch')
        self._s_mach = conn.add_stream(getattr, flight_srf, 'mach')
        self._s_q = conn.add_stream(getattr, flight_srf, 'dynamic_pressure')
        self._log_lock = threading.Lock()
        self._csv_file = None
        self._csv_writer = None
        self._log_name = None
        self._act_logging = False
        self._act_log_thread = None
        self._meco_flag = False
        self._log_t0 = None

        # Per-vessel drag table swept on the pad by sample_vessel_aero.py —
        # the game's own aero answers, so no in-flight CdA calibration.
        (self.aero_alt_grid, self.aero_speed_grid,
         self.aero_aoa_grid, self.cda_table) = load_vessel_aero_table()

        self.active_trajectory = None
        self.trajectory_lock = threading.Lock()
        self.optimizer_running = False

        self._opt_thread = None
        self._precompute_thread = None
        self._last_solution = None

        # Compile the numba sim now (background) so the first optimization
        # doesn't pay the ~5-15s JIT cost during ascent.
        self._warmup_thread = threading.Thread(target=self._warmup_jit, daemon=True)
        self._warmup_thread.start()

        # ── Delta-V tracking ────────────────────────────────────────
        # Captured here at construction — slightly after engine
        # ignition in basicGravityTurn.py, so a close but not exact
        # proxy for true pad mass.
        self._dv_lock = threading.Lock()
        self._actual_ascent_dv = 0.0
        self._dv_tracking_active = True
        self._dv_thread = threading.Thread(target=self._dv_tracker_worker, daemon=True)
        self._dv_thread.start()

    def _warmup_jit(self):
        """Trigger numba compilation of the whole sim call-chain with dummy
        inputs. Runs in a background thread at construction; numba's compile
        lock makes any optimizer thread arriving mid-compile simply wait."""
        try:
            t0 = time.time()
            r0 = np.array([self.radius_kerbin + 100.0, 0.0, 0.0])
            v0 = np.array([0.0, 0.0, 10.0])
            jit_sim_with_coast(
                r0, v0, 1000.0, 5000.0, 4000.0, 10.0, 60000.0, 70000.0,
                self.radius_kerbin, self.GM, self.omega_kerbin,
                100.0, 90.0, -30.0, 0.0, 0.0, 80000.0,
                1.0, 5.0, self.alt_table, self.rho_table, self.p_table,
                self.aero_alt_grid, self.aero_speed_grid, self.aero_aoa_grid,
                self.cda_table,
            )
            jit_sim_vertical_climb(
                r0, v0, 5000.0, 1000.0, 10.0, 60000.0, 70000.0,
                self.radius_kerbin, self.GM, self.omega_kerbin,
                300.0, 0.05, 5.0, self.alt_table, self.rho_table, self.p_table,
                self.aero_alt_grid, self.aero_speed_grid, self.aero_aoa_grid,
                self.cda_table,
            )
            jit_sim_with_coast_logged(
                r0, v0, 1000.0, 5000.0, 4000.0, 10.0, 60000.0, 70000.0,
                self.radius_kerbin, self.GM, self.omega_kerbin,
                100.0, 90.0, -30.0, 0.0, 0.0, 80000.0,
                1.0, 5.0, self.alt_table, self.rho_table, self.p_table,
                self.aero_alt_grid, self.aero_speed_grid, self.aero_aoa_grid,
                self.cda_table, self.sos_table,
                np.zeros((16, N_LOG_COLS), dtype=np.float64),
            )
            print(f"[WARMUP] numba sim compiled in {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"[WARMUP] JIT warmup failed: {e}")


    def _project_vertical_ascent_end(self, pad_state, vertical_ascent_altitude):
        """Physics-project vessel state at end of vertical_ascent, using
        pure vertical thrust (matching what vertical_ascent.py commands).
        """
        r_start = pad_state["position"][0].astype(np.float64)
        v_start = pad_state["velocity"][0].astype(np.float64)
        m_start = float(pad_state["mass"])
        mass_flow_rate = float(pad_state["mass_flow_rate"])
        max_sl_thrust = float(pad_state["max_sl_thrust"])
        max_vac_thrust = float(pad_state["max_vac_thrust"])

        r_proj, v_proj, m_proj, t_proj, climb_dv = jit_sim_vertical_climb(
            r_start,
            v_start,
            m_start,
            pad_state["fuel_mass"],
            mass_flow_rate,
            max_sl_thrust,
            max_vac_thrust,
            self.radius_kerbin,
            self.GM,
            self.omega_kerbin,
            vertical_ascent_altitude,
            dt=0.05,
            max_time=120.0,
            alt_table=self.alt_table,
            rho_table=self.rho_table,
            p_table=self.p_table,
            aero_alt_grid=self.aero_alt_grid,
            aero_speed_grid=self.aero_speed_grid,
            aero_aoa_grid=self.aero_aoa_grid,
            cda_table=self.cda_table,
        )

        omega_vec = self.omega_kerbin * np.array([0.0, 1.0, 0.0])
        v_air = v_proj + np.cross(omega_vec, r_proj)
        v_air_speed = float(np.linalg.norm(v_air))

        return {
            "position": [r_proj, float(np.linalg.norm(r_proj))],
            "velocity": [v_proj, float(np.linalg.norm(v_proj))],
            "v_air_speed": v_air_speed,
            "max_sl_thrust": max_sl_thrust,
            "max_vac_thrust": max_vac_thrust,
            "thrust_direction": pad_state["thrust_direction"],
            "exhaust_velocity": pad_state["exhaust_velocity"],
            "mass_flow_rate": mass_flow_rate,
            "rho": pad_state["rho"],
            "mass": float(m_proj),
            "fuel_mass": max(0.0, pad_state["fuel_mass"] - (m_start - m_proj)),
            "actual_pitch": 90.0,
            # ΔV already spent in the projected vertical climb, so predicted
            # ascent ΔV can cover ignition→MECO like the live tracker does.
            "climb_dv": float(climb_dv),
        }

    def begin_live_optimize(self, target_apoapsis, vertical_ascent_altitude=300.0):
        print(f"[LIVE OPT] Spawning background live optimization thread...")
        self.optimizer_running = True
        self._precompute_thread = threading.Thread(
            target=self._live_optimize_worker,
            args=(target_apoapsis, vertical_ascent_altitude),
            daemon=True
        )
        self._precompute_thread.start()

    def _live_optimize_worker(self, target_apoapsis, vertical_ascent_altitude):
        try:
            # No CdA calibration needed — the drag table was swept on the pad
            # by sample_vessel_aero.py. Snapshot immediately so the optimizer
            # gets the maximum head start over the vertical ascent.
            live_state = self.Step_1n2()
            print(f"[LIVE OPT] Live state snapshot taken at altitude {live_state['position'][1] - self.radius_kerbin:.1f}m")

            # vertical_ascent.py triggers on surface_altitude (above TERRAIN)
            # while the sim's altitude is ASL — project to the same physical
            # point by adding the pad's terrain elevation (~70 m at KSC).
            fl = self.vessel.flight()
            terrain_asl = fl.mean_altitude - fl.surface_altitude
            target_asl = vertical_ascent_altitude + max(0.0, terrain_asl)

            projected_state = self._project_vertical_ascent_end(live_state, target_asl)
            proj_alt = float(np.linalg.norm(projected_state["position"][0]) - self.radius_kerbin)
            if proj_alt < target_asl - 50.0:
                print(f"[LIVE OPT] ERROR: vertical-climb projection stalled at "
                      f"{proj_alt:.0f} m ASL (target {target_asl:.0f} m) — fuel model "
                      f"is likely wrong for this vessel (scoped fuel "
                      f"{live_state['fuel_mass']:.0f} kg). REFUSING to optimize a "
                      f"dead state; fix staging/fuel scoping and relaunch.")
                return
            self.projected_state = projected_state  # kept for the predicted trace

            print("[LIVE OPT] Beginning differential evolution optimization...")
            result = self.optimize(projected_state, target_apoapsis)
            if result is not None:
                c0, c1, c2, c3, apo_meco_trigger, predicted_final_apo, predicted_burn_time, pred_final_hvel, start_alt, predicted_ascent_dv, predicted_circ_dv, pred_meco_alt = result
                with self.trajectory_lock:
                    self.active_trajectory = {
                        "coeffs": (c0, c1, c2, c3),
                        
                        "pred_final_hvel": pred_final_hvel,
                        "start_time": None,
                        "apo_meco_trigger": apo_meco_trigger,
                        "predicted_final_apo": predicted_final_apo,
                        "predicted_burn_time": predicted_burn_time,
                        "start_alt": start_alt,
                        "predicted_ascent_dv": predicted_ascent_dv,
                        "predicted_circ_dv": predicted_circ_dv,
                        "pred_meco_alt": pred_meco_alt,
                    }
                print(f"[LIVE OPT] Trajectory optimization complete.")
            else:
                print("[LIVE OPT] Optimization failed to find a valid trajectory.")
        except Exception as e:
            print(f"[LIVE OPT] Error in live optimize thread: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.optimizer_running = False

    # ──────────────────────────────────────────────────────────────────
    # Periodic re-optimization — DISABLED BY DESIGN. The project contract
    # is ONE pre-launch optimization flown open-loop (pitch-vs-altitude):
    # the end goal is training an RL agent inside this sim, so the flight
    # exists to validate the sim's a-priori accuracy, and in-flight
    # re-optimization would mask model error. Re-enable the call site in
    # run_program's main loop only as a debugging experiment.
    # ──────────────────────────────────────────────────────────────────


    def optimizer_worker(self, current_state, target_apoapsis, opt_start_time):
        try:
            result = self.optimize(current_state, target_apoapsis)
            if result is not None:
                c0, c1, c2, c3, apo_meco_trigger, predicted_final_apo, predicted_burn_time, pred_final_hvel, start_alt, predicted_ascent_dv, predicted_circ_dv, pred_meco_alt = result
                with self.trajectory_lock:
                    self.active_trajectory = {
                        "coeffs": (c0, c1, c2, c3),
                        
                        "pred_final_hvel": pred_final_hvel,
                        "start_time": opt_start_time,
                        "apo_meco_trigger": apo_meco_trigger,
                        "predicted_final_apo": predicted_final_apo,
                        "predicted_burn_time": predicted_burn_time,
                        "start_alt": start_alt,
                        "predicted_ascent_dv": predicted_ascent_dv,
                        "predicted_circ_dv": predicted_circ_dv,
                        "pred_meco_alt": pred_meco_alt,
                    }
                print(f"  [OPT] Trajectory updated — "
                      f"c0={c0:.1f}° c1={c1:.4f} c2={c2:.6f} c3={c3:.8f} "
                      f"MECO@{apo_meco_trigger:.0f}m PredFinalApo={predicted_final_apo:.0f}m")
        except Exception as e:
            print(f"  [OPT] Optimizer error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.optimizer_running = False

    # ──────────────────────────────────────────────────────────────────
    # Main control loop
    # ──────────────────────────────────────────────────────────────────

    def run_program(self, target_apoapsis):
        # Anchor the burn clock HERE — run_program is entered the moment
        # vertical ascent ends (~300 m), which is the same t=0 the sim's
        # predicted_burn_time uses (projection to vertical_ascent_altitude).
        # Setting it after the optimizer wait made actual burns read short.
        burn_start_time = self.conn.space_center.ut
        self.vessel.auto_pilot.reference_frame = self.vessel.surface_reference_frame
        self.vessel.auto_pilot.target_pitch_and_heading(89.5, 90)
        self.vessel.auto_pilot.target_roll = 0.0
        dt_refresh = 0.1

        # else fall back to a blocking optimization right now.

        with self.trajectory_lock:
            traj = self.active_trajectory

        if traj is None:
            if self.optimizer_running:
                print("[RUN] Waiting on live optimization...")
                while self.active_trajectory is None and self.optimizer_running:
                    time.sleep(0.1)
                with self.trajectory_lock:
                    traj = self.active_trajectory
                    if traj is None:
                        print("[INIT] Optimizer failed — aborting.")
                        return
            else:
                print("[INIT] No precomputed trajectory — running blocking optimization…")
                current_state = self.Step_1n2()
                if current_state is None:
                    print("[INIT] Could not read initial state — aborting.")
                    return
                opt_start = time.time()
                init_result = self.optimize(current_state, target_apoapsis)
                if init_result is None:
                    print("[INIT] Optimizer failed — aborting.")
                    return
                c0, c1, c2, c3, apo_meco_trigger, predicted_final_apo, predicted_burn_time, pred_final_hvel, start_alt, predicted_ascent_dv, predicted_circ_dv, pred_meco_alt = init_result
                with self.trajectory_lock:
                    self.active_trajectory = {
                        "coeffs": (c0, c1, c2, c3),
                        
                        "pred_final_hvel": pred_final_hvel,
                        "start_time": opt_start,
                        "apo_meco_trigger": apo_meco_trigger,
                        "predicted_final_apo": predicted_final_apo,
                        "predicted_burn_time": predicted_burn_time,
                        "start_alt": start_alt,
                        "predicted_ascent_dv": predicted_ascent_dv,
                        "predicted_circ_dv": predicted_circ_dv,
                        "pred_meco_alt": pred_meco_alt,
                    }
                    traj = self.active_trajectory
        
        if traj.get("start_time") is None:
            # Precomputed trajectory — anchor t=0 to right now, the real
            # moment the gravity turn begins.
            with self.trajectory_lock:
                self.active_trajectory["start_time"] = time.time()
                traj = self.active_trajectory
            print("[RUN] Using precomputed trajectory (anchored at t=0 now).")

        print(
            f"[RUN] Trajectory — c0..c3={traj['coeffs']} "
            f"MECO@{traj['apo_meco_trigger']:.0f}m "
            f"PredFinalApo={traj['predicted_final_apo']:.0f}m "
            f"PredBurn={traj['predicted_burn_time']:.1f}s"
        )
        
        print("\n=== VERTICAL ASCENT COMPLETED ===")
        print("PREDICTED STATE (from RK4 projection):")
        if hasattr(self, 'projected_state'):
            for k, v in self.projected_state.items():
                if isinstance(v, (list, np.ndarray)):
                    print(f"  {k}: {v}")
                elif isinstance(v, float):
                    print(f"  {k}: {v:.6f}")
                else:
                    print(f"  {k}: {v}")
        print("\nACTUAL STATE (telemetry right now):")
        self.Step_1n2()
        print("=================================\n")

        # ── Flight log: write the predicted trace, then stream actuals ──
        self._start_flight_log(traj, burn_start_time)

        last_opt_launch = time.time()
        meco_has_occurred = False
        actual_burn_time = 0.0

        # ── Main loop ────────────────────────────────────────────────
        while True:
            altitude = self.vessel.flight(self.ref).mean_altitude
            
            # Exit loop completely ONLY when we hit space (70km) AND engine is off
            if altitude >= 70000.0 and meco_has_occurred:
                break
                
            self.check_staging()

            with self.trajectory_lock:
                traj = self.active_trajectory

            # Trigger MECO but KEEP FLYING to control coast pitch
            if not meco_has_occurred and self.apoapsis() >= traj["apo_meco_trigger"]:
                meco_has_occurred = True
                self._meco_flag = True
                self.vessel.control.throttle = 0.0
                self.stop_dv_tracking()
                actual_burn_time = self.conn.space_center.ut - burn_start_time
                print(
                    f"\n*** MECO — engine cut-off ***\n"
                    f"Predicted Burn: {traj['predicted_burn_time']:.1f}s | "
                    f"Actual Burn: {actual_burn_time:.1f}s | "
                    f"Error: {actual_burn_time - traj['predicted_burn_time']:+.1f}s\n"
                )

            c0, c1, c2, c3 = traj["coeffs"]
            
            start_alt = traj["start_alt"]
            
            dh = (altitude - start_alt) / 40000.0
            if dh < 0.0: dh = 0.0
            
            pitch_command = c0 + c1 * dh + c2 * dh**2 + c3 * dh**3
            pitch_command = float(np.clip(pitch_command, 0, 90))
            if meco_has_occurred:
                # Coast attitude: surface prograde (AoA≈0, minimum drag) —
                # matches the sim's post-MECO drag model. The cubic past MECO
                # altitude is unfitted extrapolation, not a trajectory choice.
                # Must use the body's ROTATING frame: the no-arg flight()
                # frame moves with the vessel, so its speeds read ~0 and the
                # command collapses to pitch 0.
                srf = self.vessel.flight(self.vessel.orbit.body.reference_frame)
                pitch_command = float(np.clip(math.degrees(math.atan2(
                    srf.vertical_speed, max(1e-6, srf.horizontal_speed))), 0, 90))
            
            # Only throttle if we haven't hit MECO yet
            if not meco_has_occurred:
                self.vessel.control.throttle = 1.0

            self.vessel.auto_pilot.target_pitch_and_heading(pitch_command, 90)
            self.vessel.auto_pilot.target_roll = 0.0

            flight = self.vessel.flight(self.ref)
            # Inertial-frame horizontal speed, same frame as pred_final_hvel
            # (the rotating-frame value reads ~175 m/s lower at the equator).
            v_horiz = flight.horizontal_speed
            
            mode_str = "COAST" if meco_has_occurred else "BURN "
            print(f"[{mode_str}] Alt: {self.surface_altitude():.0f}m | "
                  f"Apo: {self.apoapsis():.0f}m | "
                  f"Vel: {flight.speed:.0f}m/s (H: {v_horiz:.0f}m/s) | "
                  f"Cmd: {pitch_command:.1f}° | Act: {self.vessel.flight().pitch:.1f}° | "
                  f"MECO@{traj['apo_meco_trigger']:.0f}m (Alt: {traj['pred_meco_alt']:.0f}m) | "
                  f"Pred H-Vel@{traj['pred_final_hvel']:.0f}m/s")

            # Single-optimization contract: this call site stays disabled —
            # the whole ascent flies the pre-launch trajectory (see the

            time.sleep(dt_refresh)

        actual_ascent_dv = self.get_actual_ascent_dv()

        # ── Validate: watch the real apoapsis settle, compare to predicted
        with self.trajectory_lock:
            predicted_final_apo = self.active_trajectory.get("predicted_final_apo")

        if predicted_final_apo is not None:
            print(f"[VALIDATE] Watching real apoapsis settle "
                  f"(predicted = {predicted_final_apo:.0f}m)…")
            last_apo = None
            stable_count = 0
            t_watch = 0.0
            while t_watch < 90.0 and stable_count < 5:
                apo_now = self.apoapsis()
                if last_apo is not None and abs(apo_now - last_apo) < 5.0:
                    stable_count += 1
                else:
                    stable_count = 0
                last_apo = apo_now
                time.sleep(1.0)
                t_watch += 1.0
            print(f"[VALIDATE] Real settled apoapsis ≈ {last_apo:.0f}m | "
                  f"Predicted = {predicted_final_apo:.0f}m | "
                  f"Error = {last_apo - predicted_final_apo:+.0f}m")
            
            
            # ── Delta-V comparison: actual vs predicted ──────────────
            r_apo_actual = self.vessel.orbit.apoapsis_altitude + self.radius_kerbin
            r_peri_actual = self.vessel.orbit.periapsis_altitude + self.radius_kerbin
            a_actual = (r_apo_actual + r_peri_actual) / 2.0
            v_apo_actual = math.sqrt(self.GM * (2.0 / r_apo_actual - 1.0 / a_actual))
            v_circular_actual = math.sqrt(self.GM / r_apo_actual)
            actual_circ_dv = abs(v_circular_actual - v_apo_actual)
            actual_total_dv = actual_ascent_dv + actual_circ_dv

            with self.trajectory_lock:
                predicted_ascent_dv = self.active_trajectory.get("predicted_ascent_dv")
                predicted_circ_dv = self.active_trajectory.get("predicted_circ_dv")
            predicted_total_dv = (predicted_ascent_dv or 0.0) + (predicted_circ_dv or 0.0)

            print("\n=== DELTA-V REPORT ===")
            print(f"Ascent ΔV          — Actual: {actual_ascent_dv:.1f} m/s | Predicted: {predicted_ascent_dv:.1f} m/s")
            print(f"Circularization ΔV — Actual: {actual_circ_dv:.1f} m/s | Predicted: {predicted_circ_dv:.1f} m/s  (both estimated, no burn performed)")
            print(f"TOTAL mission ΔV   — Actual: {actual_total_dv:.1f} m/s | Predicted: {predicted_total_dv:.1f} m/s")
            print("=======================\n")

        # ── Join any in-flight background threads before exiting, to
        # avoid the daemon-thread-print-at-shutdown crash
        for th in (self._opt_thread, self._precompute_thread):
            if th is not None and th.is_alive():
                th.join(timeout=5.0)

        self._stop_flight_log()

    # ──────────────────────────────────────────────────────────────────
    # Flight logging (predicted trace + live actual telemetry)
    # ──────────────────────────────────────────────────────────────────

    def _start_flight_log(self, traj, t0_ut):
        self._log_name = time.strftime("flight_log_%Y%m%d_%H%M%S.csv")
        self._csv_file = open(self._log_name, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(LOG_COLUMNS)
        try:
            n = self._write_predicted_trace(traj)
            print(f"[LOG] Predicted trace written ({n} rows) → {self._log_name}")
        except Exception as e:
            print(f"[LOG] Could not write predicted trace: {e}")
        self._csv_file.flush()
        self._log_t0 = t0_ut
        self._act_logging = True
        self._act_log_thread = threading.Thread(
            target=self._actual_log_worker, daemon=True
        )
        self._act_log_thread.start()

    def _write_predicted_trace(self, traj):
        """Re-run the chosen trajectory once through the logged sim from the
        same projected 300 m state the optimizer used, and dump every step."""
        if getattr(self, "projected_state", None) is None:
            print("[LOG] No projected state stored — predicted trace skipped.")
            return 0
        st = self.projected_state
        c0, c1, c2, c3 = traj["coeffs"]
        r0 = st["position"][0].astype(np.float64)
        v0 = st["velocity"][0].astype(np.float64)
        fuel = float(st["fuel_mass"])
        m0 = float(st["mass"])
        log_out = np.zeros((20000, N_LOG_COLS), dtype=np.float64)
        _, _, _, _, n_rows = jit_sim_with_coast_logged(
            r0, v0, fuel, m0, m0 - fuel,
            float(st["mass_flow_rate"]), float(st["max_sl_thrust"]),
            float(st["max_vac_thrust"]),
            self.radius_kerbin, self.GM, self.omega_kerbin,
            float(traj["start_alt"]), c0, c1, c2, c3,
            float(traj["apo_meco_trigger"]),
            1.0, 1200.0, self.alt_table, self.rho_table, self.p_table,
            self.aero_alt_grid, self.aero_speed_grid, self.aero_aoa_grid,
            self.cda_table, self.sos_table, log_out,
        )
        with self._log_lock:
            for i in range(n_rows):
                self._csv_writer.writerow(
                    ["pred"] + [f"{x:.6g}" for x in log_out[i]]
                )
        return int(n_rows)

    def _actual_log_worker(self):
        """5 Hz kRPC telemetry sampler — same columns and t=0 as the
        predicted trace. All reads are streams (local cache, no RPC cost)."""
        while self._act_logging:
            try:
                with self.trajectory_lock:
                    traj = self.active_trajectory
                t = self.ut() - self._log_t0
                alt = self._s_mean_alt()
                pos = np.array(self.position())
                rr = float(np.linalg.norm(pos))
                rho = self.rho()
                vair = self._s_tas()
                try:
                    drag = float(np.linalg.norm(np.array(self._s_drag())))
                except Exception:
                    drag = float('nan')
                if rho > 1e-7 and vair > 30.0 and math.isfinite(drag):
                    cda = 2.0 * drag / (rho * vair * vair)
                else:
                    cda = float('nan')
                if traj is not None:
                    c0, c1, c2, c3 = traj["coeffs"]
                    dh = max(0.0, (alt - traj["start_alt"]) / 40000.0)
                    pitch_cmd = float(np.clip(
                        c0 + c1 * dh + c2 * dh**2 + c3 * dh**3, 0.0, 90.0
                    ))
                else:
                    pitch_cmd = float('nan')
                row = [
                    t,
                    0.0 if self._meco_flag else 1.0,
                    alt,
                    self._s_speed(),
                    self._s_hspeed(),
                    self._s_vspeed(),
                    self.apoapsis(),
                    self.mass(),
                    self.thrust(),
                    self.g_force() * 9.80665,
                    self.GM / (rr * rr),
                    drag,
                    cda,
                    self._s_aoa(),
                    self._s_pitch(),
                    pitch_cmd,
                    self._s_mach(),
                    self._s_q(),
                    rho,
                ]
                with self._log_lock:
                    self._csv_writer.writerow(
                        ["act"] + [f"{x:.6g}" for x in row]
                    )
                    self._csv_file.flush()
            except Exception:
                pass
            time.sleep(0.2)

    def _stop_flight_log(self):
        self._act_logging = False
        if self._act_log_thread is not None and self._act_log_thread.is_alive():
            self._act_log_thread.join(timeout=2.0)
        if self._csv_file is not None:
            with self._log_lock:
                self._csv_file.flush()
                self._csv_file.close()
                self._csv_file = None
            print(f"[LOG] Flight log saved → {self._log_name}")

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def check_staging(self):
        """First-stage-only contract: the program NEVER activates stages.
        Stage-1 flameout just means the engine is done — detect and warn only.
        (The old auto-stager also fired spuriously right after ignition, when
        has_fuel briefly reads False, detaching the first stage on
        multi-stage rockets.)"""
        now = time.time()
        if now - getattr(self, "_last_stage_check", 0.0) < 0.5:
            return
        self._last_stage_check = now
        if getattr(self, "_flameout_warned", False):
            return
        if not hasattr(self, "_engines_cache"):
            self._engines_cache = self.vessel.parts.engines
        for engine in self._engines_cache:
            if engine.active and not engine.has_fuel:
                self._flameout_warned = True
                print("[WARN] stage-1 flameout — coasting ballistic "
                      "(auto-staging disabled: first-stage-only contract)")
                return

    def stage_fuel_mass(self):
        """First-stage propellant: LiquidFuel+Oxidizer in every part that
        detaches WITH or BEFORE the active engines (decouple_stage >= the
        engines'), so drop tanks count and upper stages stay dead weight.
        Never-detaching parts report decouple_stage -1, so a single-stick
        rocket sums the whole craft. Serial staging assumed; boosters feeding
        a never-detaching core engine will undercount (out of scope)."""
        try:
            engines = [e for e in self.vessel.parts.engines if e.active]
            if not engines:
                engines = self.vessel.parts.engines
            s = max(e.part.decouple_stage for e in engines)
            total = 0.0
            for st in range(s, self.vessel.control.current_stage + 1):
                res = self.vessel.resources_in_decouple_stage(st, cumulative=False)
                total += (res.amount('LiquidFuel') + res.amount('Oxidizer')) * 5.0
            print(f"[FUEL] stage-scoped propellant: {total:.0f} kg "
                  f"(parts decoupling at stage >= {s})")
            return total
        except Exception:
            print("[WARN] stage-scoped fuel lookup failed — falling back to "
                  "WHOLE-VESSEL propellant (overcounts on multi-stage rockets)")
            lf = self.vessel.resources.amount('LiquidFuel')
            ox = self.vessel.resources.amount('Oxidizer')
            return (lf + ox) * 5.0

    # ──────────────────────────────────────────────────────────────────
    # State snapshot
    # ──────────────────────────────────────────────────────────────────

    def Step_1n2(self):
        position = np.array(self.position())
        print('pos: ', position)
        velocity = np.array(self.velocity())
        print('vel: ', velocity)
        omega_vec = self.omega_kerbin * np.array([0.0, 1.0, 0.0])
        print('omega_vec: ', omega_vec)
        v_air = velocity + np.cross(omega_vec, position)
        print('v_air: ', v_air)
        v_air_speed = float(np.linalg.norm(v_air))
        print('v_air_speed: ', v_air_speed)
        # Frame-handedness tripwire: v_air is the surface-frame velocity, so its
        # magnitude must match KSP's own airspeed reading. A big gap means the
        # ω×r sign regressed (left-handed kRPC frame vs right-handed np.cross).
        try:
            tas = float(self._s_tas())
            if abs(v_air_speed - tas) > 20.0:
                print(f"[FRAME WARNING] sim v_air_speed {v_air_speed:.1f} m/s vs "
                      f"kRPC true_air_speed {tas:.1f} m/s — ω×r sign regression?")
        except Exception:
            pass
        actual_thrust = self.thrust()
        print('actual_thrust: ', actual_thrust)
        sensed_accel = self.g_force() * 9.80665
        print('sensed_accel: ', sensed_accel)
        
        # Protect against negative drag calculations
        drag_force = actual_thrust - self.mass() * sensed_accel
        drag_force = max(0.0, min(drag_force, actual_thrust))
        print('drag_force: ', drag_force)
        
        thrust_magnitude = actual_thrust
        print('thrust_magnitude: ', thrust_magnitude)
        thrust_direction = np.array(self.direction())
        print('thrust_direction: ', thrust_direction)
        exhaust_velocity = 9.80665 * self.isp()
        print('exhaust_velocity: ', exhaust_velocity)
        
        engines = self.vessel.parts.engines
        active_engines = [e for e in engines if e.active]
        
        # --- ON-PAD LOOKAHEAD LOGIC ---
        if not active_engines:
            target_stage = max((e.part.stage for e in engines), default=-1)
            active_engines = [e for e in engines if e.part.stage == target_stage]
            drag_force = 0.0
        
        max_sl_thrust = 0.0
        max_vac_thrust = 0.0
        mass_flow_rate = 0.0
        
        for engine in active_engines:
            limit = engine.thrust_limit
            # max_thrust_at(pressure_atm) is unambiguous at any altitude —
            # engine.max_thrust would be thrust at CURRENT pressure, which the
            # sim would then double-correct when re-optimizing mid-flight.
            max_sl_thrust += engine.max_thrust_at(1.0) * limit
            vac_t = engine.max_thrust_at(0.0)
            max_vac_thrust += vac_t * limit
            isp = engine.vacuum_specific_impulse
            if isp > 0:
                mass_flow_rate += (vac_t * limit) / (isp * 9.80665)
        
        if mass_flow_rate > 0:
            exhaust_velocity = max_vac_thrust / mass_flow_rate
        else:
            return None
            
        print('mass_flow_rate: ', mass_flow_rate)
        actual_pitch = self.vessel.flight().pitch
        print('actual_pitch: ', actual_pitch)
        return {
            "position": [position, np.linalg.norm(position)],
            "velocity": [velocity, np.linalg.norm(velocity)],
            "v_air_speed": v_air_speed,
            "max_sl_thrust": max_sl_thrust,
            "max_vac_thrust": max_vac_thrust,
            "thrust_direction": thrust_direction,
            "exhaust_velocity": exhaust_velocity,
            "mass_flow_rate": mass_flow_rate,
            "rho": self.rho(),
            "mass": self.mass(),
            "fuel_mass": self.stage_fuel_mass(),
            "actual_pitch": actual_pitch,
            "actual_throttle": self.vessel.control.throttle,
        }

    # ──────────────────────────────────────────────────────────────────
    # Simulation & Optimizer
    # ──────────────────────────────────────────────────────────────────

    def run_sim_from_state(self, current_state, c0, c1, c2, c3, apo_meco_trigger):
        r_start = current_state["position"][0].astype(np.float64)
        v_start = current_state["velocity"][0].astype(np.float64)
        fuel_mass = float(current_state["fuel_mass"])
        m_start = float(current_state["mass"])
        dry_mass = m_start - fuel_mass
        mass_flow_rate = float(current_state["mass_flow_rate"])
        max_sl_thrust = float(current_state["max_sl_thrust"])
        max_vac_thrust = float(current_state["max_vac_thrust"])
        start_alt = float(np.linalg.norm(r_start) - self.radius_kerbin)

        r_final, v_final, m_final, meco_occurred, meco_time, max_load, meco_alt, predicted_ascent_dv = jit_sim_with_coast(
            r_start, v_start, fuel_mass, m_start, dry_mass, mass_flow_rate,
            max_sl_thrust, max_vac_thrust, self.radius_kerbin,
            self.GM, self.omega_kerbin,
            start_alt, c0, c1, c2, c3, apo_meco_trigger, dt_step=1.0, max_time=1200.0,
            alt_table=self.alt_table, rho_table=self.rho_table, p_table=self.p_table,
            aero_alt_grid=self.aero_alt_grid, aero_speed_grid=self.aero_speed_grid,
            aero_aoa_grid=self.aero_aoa_grid, cda_table=self.cda_table,
        )

        return {
            "position": r_final,
            "velocity": v_final,
            "mass": m_final,
            "meco_occurred": meco_occurred,
            "meco_time": meco_time,
            "max_load": max_load,
            "meco_alt": meco_alt,
            "predicted_ascent_dv": predicted_ascent_dv,
        }

    def cost_function(self, params, current_state, target_apoapsis, c0):
        c1, c2, c3, apo_overshoot = params
        apo_meco_trigger = target_apoapsis + apo_overshoot

        final = self.run_sim_from_state(
            current_state, c0, c1, c2, c3, apo_meco_trigger
        )

        if not final["meco_occurred"]:
            return 1e10

        post_coast_apo = jit_compute_apoapsis(
            final["position"], final["velocity"],
            self.GM, self.radius_kerbin
        )
        if not math.isfinite(post_coast_apo) or post_coast_apo <= 0:
            return 1e10

        m_meco = final["mass"]
        r_apo = post_coast_apo + self.radius_kerbin
        v_circular = math.sqrt(self.GM / r_apo)

        h_mag = np.linalg.norm(np.cross(final["position"], final["velocity"]))
        v_at_apo = h_mag / r_apo

        dv_circ = abs(v_circular - v_at_apo)

        ve = current_state["exhaust_velocity"]
        m_after_circ = m_meco * math.exp(-dv_circ / ve)
        total_fuel = current_state["mass"] - m_after_circ
        max_load = final["max_load"]

        # Smooth physics based penalty to avoid destroying the optimizer space
        # with a hard cliff. Threshold is in REAL q*AoA units: flown gravity
        # turns sit around 1.5e5-5e5 Pa·deg. (The old 4e4 threshold was tuned
        # against the pre-frame-fix phantom AoA and made every real turn look
        # structurally infeasible — the optimizer answered with vertical lobs.)
        load_penalty = 0.0
        if max_load > 300000.0:
            load_penalty = ((max_load - 300000.0) / 15000.0) ** 2

        # Circularization must be affordable WITH margin: the impulsive circ
        # estimate is optimistic (finite burn, steering losses), so a plan that
        # "just fits" on paper runs dry in the real flight.
        reserve_dv = 75.0  # m/s of ΔV headroom the plan must keep after circ
        dry_mass = float(current_state["mass"]) - float(current_state["fuel_mass"])
        fuel_at_meco = max(0.0, m_meco - dry_mass)
        m_after_reserve = m_after_circ * math.exp(-reserve_dv / ve)
        circ_fuel_needed = m_meco - m_after_reserve
        reserve_penalty = 0.0
        deficit = circ_fuel_needed - fuel_at_meco
        if deficit > 0.0:
            reserve_penalty = 1e4 + 100.0 * deficit

        apo_error_km = abs(post_coast_apo - target_apoapsis) / 1000.0
        return total_fuel + 50.0 * (apo_error_km ** 2) + load_penalty + reserve_penalty

    def optimize(self, current_state, target_apoapsis):
        c0 = float(current_state["actual_pitch"])
        c0 = max(5.0, min(90.0, c0))

        bounds = [
            (-220.0,  0.0),       # c1 (pitch should generally decrease)
            (-100.0,  180.0),     # c2
            (-80.0,   80.0),      # c3
            (0.0,     30000.0),   # apo_overshoot
        ]

        def cost(x):
            c1, c2, c3, apo_overshoot = x
            return self.cost_function((c1, c2, c3, apo_overshoot), current_state, target_apoapsis, c0)

        # Warm start: seed the initial population with the previous solution
        # so re-optimizations converge in a fraction of the iterations.
        popsize = 15
        n_individuals = popsize * len(bounds)
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        init_pop = lo + rng.random((n_individuals, len(bounds))) * (hi - lo)
        if self._last_solution is not None:
            init_pop[0] = np.clip(self._last_solution, lo, hi)
        # Always seed one canonical gravity-turn shape (pitch 90→0 across the
        # dh range) so the population contains an orbital-insertion basin
        # member from iteration zero, warm start or not.
        init_pop[1] = np.clip(np.array([-140.0, 55.0, -5.0, 1000.0]), lo, hi)

        t_opt0 = time.time()
        result = differential_evolution(
            cost, bounds=bounds, maxiter=150, tol=1e-3, seed=42,
            polish=False, init=init_pop,
        )
        print(f"  Optimizer: solved in {time.time() - t_opt0:.1f}s "
              f"({result.nfev} sims, {result.nit} iterations)")
        self._last_solution = result.x.copy()

        c1, c2, c3, apo_overshoot = result.x
        apo_meco_trigger = target_apoapsis + apo_overshoot

        final = self.run_sim_from_state(
            current_state, c0, c1, c2, c3, apo_meco_trigger
        )
        if not final["meco_occurred"]:
            print("  Optimizer: MECO not reached (fuel exhaustion?)")
            return None

        post_apo = jit_compute_apoapsis(
            final["position"], final["velocity"],
            self.GM, self.radius_kerbin
        )

        ve = current_state["exhaust_velocity"]
        # Predicted Δv for the WHOLE ascent: vertical climb (projected)
        # plus gravity turn to MECO — same span the live tracker measures.
        predicted_ascent_dv = final["predicted_ascent_dv"] + current_state.get("climb_dv", 0.0)

        # Predicted circularization Δv — identical formula to the one
        # already inside cost_function.
        r_apo_pred = post_apo + self.radius_kerbin
        v_circular_pred = math.sqrt(self.GM / r_apo_pred)
        h_mag_pred = np.linalg.norm(np.cross(final["position"], final["velocity"]))
        v_at_apo_pred = h_mag_pred / r_apo_pred
        predicted_circ_dv = abs(v_circular_pred - v_at_apo_pred)


        print(f"  Optimizer: Fuel={result.fun:.1f}kg | "
              f"PostCoastApo={post_apo:.0f}m | MECO@{apo_meco_trigger:.0f}m | "
              f"c0={c0:.1f}° c1={c1:.4f} c2={c2:.6f} c3={c3:.8f} | "
              f"PredAscentΔV={predicted_ascent_dv:.0f}m/s | PredCircΔV={predicted_circ_dv:.0f}m/s")

        pred_final_hvel = float(np.linalg.norm(np.cross(final["position"], final["velocity"])) / (post_apo + self.radius_kerbin))
        start_alt = float(np.linalg.norm(current_state["position"][0]) - self.radius_kerbin)
        pred_meco_alt = final["meco_alt"]
        return (
            c0,
            c1,
            c2,
            c3,
            apo_meco_trigger,
            post_apo,
            final["meco_time"],
            pred_final_hvel,
            start_alt,
            predicted_ascent_dv, 
            predicted_circ_dv,
            pred_meco_alt
        )
    
    def _dv_tracker_worker(self):
        """Accumulates ACTUAL delta-v spent via real-time Isp and mass:
            dv_increment = Isp(t) * g0 * ln(m_prev / m_now)
        Summing increments correctly handles Isp varying between
        sea-level and vacuum ratings over the real flight — more
        accurate than one constant-Isp Tsiolkovsky calc from pad mass
        straight to final mass.
        """
        prev_mass = self.mass()
        prev_t = time.time()
        while self._dv_tracking_active:
            time.sleep(0.05)
            try:
                current_mass = self.mass()
                current_isp = self.isp()
            except Exception:
                continue
            now = time.time()
            dt = max(1e-3, now - prev_t)
            prev_t = now
            if current_mass <= 0 or current_isp <= 0:
                continue
            if current_mass < prev_mass:
                # Separation events (fairings, decouplers) are step drops;
                # burning is a continuous RATE. Engines burn at most
                # a/ve ≈ 2%/s of vessel mass, so a loss rate above 4%/s is
                # a jettison — skip it. Rate-based (not per-tick fraction)
                # so ticks stretched by scheduler stalls aren't discarded.
                rel_rate = (prev_mass - current_mass) / prev_mass / dt
                if rel_rate < 0.04:
                    dv_increment = current_isp * 9.80665 * math.log(prev_mass / current_mass)
                    with self._dv_lock:
                        self._actual_ascent_dv += dv_increment
            prev_mass = current_mass

    def get_actual_ascent_dv(self):
        with self._dv_lock:
            return self._actual_ascent_dv

    def stop_dv_tracking(self):
        self._dv_tracking_active = False