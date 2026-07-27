import time
import math
import numpy as np


class Circularization():
    """Circularize at apoapsis after the gravity turn + coast.

    No maneuver node is created — everything is computed directly:
        dv     = v_circ(r_apo) - v_apo            (vis-viva)
        t_burn = m0*ve/F * (1 - e^(-dv/ve))       (rocket equation)
    Ignition starts when time-to-apoapsis equals half the burn duration,
    centering the burn on apoapsis, and the vessel points orbital prograde
    throughout. Returns the ACTUAL burned dv (vacuum Isp * mass ratio) so
    the mission's real total dv can be reported.
    """

    def __init__(self, vessel, conn):
        self.vessel = vessel
        self.conn = conn
        body = vessel.orbit.body
        self.GM = body.gravitational_parameter
        self.ref = body.non_rotating_reference_frame
        self.position = conn.add_stream(vessel.position, self.ref)
        self.velocity = conn.add_stream(vessel.velocity, self.ref)
        self.mass = conn.add_stream(getattr, vessel, 'mass')
        self.time_to_apoapsis = conn.add_stream(
            getattr, vessel.orbit, 'time_to_apoapsis'
        )

    def _vacuum_engine_stats(self):
        """Total vacuum thrust, mass flow and effective ve of active engines.
        Read from static engine properties so it works with throttle at 0."""
        thrust = 0.0
        mdot = 0.0
        for engine in self.vessel.parts.engines:
            if not engine.active:
                continue
            limit = engine.thrust_limit
            vac_t = engine.max_thrust_at(0.0) * limit
            thrust += vac_t
            isp = engine.vacuum_specific_impulse
            if isp > 0:
                mdot += vac_t / (isp * 9.80665)
        if thrust <= 0.0 or mdot <= 0.0:
            return None
        return thrust, mdot, thrust / mdot

    def run(self):
        print("\nPhase: Circularization")

        orbit = self.vessel.orbit
        r_apo = orbit.apoapsis  # radius from body center
        a = orbit.semi_major_axis
        v_apo = math.sqrt(self.GM * (2.0 / r_apo - 1.0 / a))
        v_circ = math.sqrt(self.GM / r_apo)
        dv_planned = v_circ - v_apo

        stats = self._vacuum_engine_stats()
        if stats is None:
            print("[CIRC] No active engines with thrust — aborting.")
            return 0.0
        thrust, mdot, ve = stats

        m0 = self.mass()
        t_burn = (m0 * ve / thrust) * (1.0 - math.exp(-dv_planned / ve))
        print(f"[CIRC] Apoapsis {orbit.apoapsis_altitude:.0f}m | "
              f"planned dv={dv_planned:.1f}m/s | burn time={t_burn:.1f}s | "
              f"ignition at T-apo={t_burn / 2.0:.1f}s")

        # Point orbital prograde and let the autopilot settle.
        ap = self.vessel.auto_pilot
        ap.reference_frame = self.vessel.orbital_reference_frame
        ap.target_direction = (0.0, 1.0, 0.0)
        ap.target_roll = float('nan')
        ap.engage()
        t0 = time.time()
        while time.time() - t0 < 30.0:
            d = np.array(self.vessel.direction(self.vessel.orbital_reference_frame))
            if d[1] > 0.996:  # within ~5 deg of prograde
                break
            time.sleep(0.2)

        # Warp to shortly before ignition if the coast is long.
        lead = 15.0
        ignition_in = self.time_to_apoapsis() - t_burn / 2.0
        if ignition_in > lead + 5.0:
            print(f"[CIRC] Warping through {ignition_in - lead:.0f}s of coast...")
            self.conn.space_center.warp_to(self.conn.space_center.ut + ignition_in - lead)

        while self.time_to_apoapsis() - t_burn / 2.0 > 0.05:
            time.sleep(0.05)

        print("[CIRC] Ignition.")
        m_start = self.mass()
        ut_ignition = self.conn.space_center.ut
        self.vessel.control.throttle = 1.0

        while True:
            r = float(np.linalg.norm(np.array(self.position())))
            v = float(np.linalg.norm(np.array(self.velocity())))
            dv_remaining = math.sqrt(self.GM / r) - v
            if dv_remaining <= 0.1:
                break
            if self.vessel.max_thrust <= 0.0:
                print("[CIRC] WARNING: no thrust available (out of fuel?) — stopping burn.")
                break
            # Taper the last couple of seconds so the cutoff doesn't overshoot.
            accel = self.vessel.max_thrust / self.mass()
            self.vessel.control.throttle = float(
                min(1.0, max(0.05, dv_remaining / (accel * 1.5)))
            )
            time.sleep(0.05)

        self.vessel.control.throttle = 0.0
        burn_duration = self.conn.space_center.ut - ut_ignition
        m_end = self.mass()
        dv_actual = ve * math.log(m_start / m_end)

        time.sleep(2.0)
        orbit = self.vessel.orbit
        print(f"\n=== CIRCULARIZATION REPORT ===")
        print(f"dv        — planned (impulsive): {dv_planned:.1f} m/s | actually burned: {dv_actual:.1f} m/s")
        print(f"burn time — planned: {t_burn:.1f}s | actual: {burn_duration:.1f}s")
        print(f"orbit     — apo: {orbit.apoapsis_altitude:.0f}m | peri: {orbit.periapsis_altitude:.0f}m | "
              f"ecc: {orbit.eccentricity:.4f}")
        print(f"==============================\n")
        return dv_actual
