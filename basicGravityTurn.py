from phases import vertical_ascent
import krpc
import time
import phases.vertical_ascent
import phases.pitch_program
# To record a predicted-vs-actual telemetry CSV for the flight, swap in the
# logging build: import phases.pitch_program_w_logs and construct
# phases.pitch_program_w_logs.PitchProgram(...) below instead (same API).
import phases.pitch_program_w_logs
import phases.circularization

TARGET_APOAPSIS = 80000

conn = krpc.connect(rpc_port=50010, stream_port=50011)
print("kRPC version:", conn.krpc.get_status().version)

vessel = conn.space_center.active_vessel
print("The controlling vessel:", vessel.name)


def run_flight():
    vessel.auto_pilot.engage()
    vessel.auto_pilot.target_pitch_and_heading(90, 90)
    vessel.auto_pilot.target_roll = 0.0

    # Throttle up BEFORE igniting — engines staged at zero throttle briefly
    # report has_fuel=False, which is what tripped the old auto-stager.
    vessel.control.throttle = 1.0
    vessel.control.activate_next_stage()

    # PitchProgram construction starts the JIT warmup thread, and
    # begin_live_optimize snapshots state imme,diately — both run in parallel
    # with the vertical climb, so by the time vertical_ascent.run(300)
    # returns, the first trajectory is ready or nearly ready.
    # Requires vessel_aero_table.json (run phases/sample_vessel_aero.py on
    # the pad once per rocket) and an up-to-date kerbin_atmosphere_table.json.
    pitch_program = phases.pitch_program_w_logs.PitchProgram(vessel, conn)
    pitch_program.begin_live_optimize(TARGET_APOAPSIS)

    print("Live optimization thread launched. Launching vehicle.")


    vertical_ascent = phases.vertical_ascent.VerticalAscent(vessel, conn)
    vertical_ascent.run(300)

    pitch_program.run_program(TARGET_APOAPSIS)

    circularization = phases.circularization.Circularization(vessel, conn)
    circ_dv = circularization.run()

    ascent_dv = pitch_program.get_actual_ascent_dv()
    print(f"=== FLOWN MISSION ΔV: ascent {ascent_dv:.1f} + "
          f"circularization {circ_dv:.1f} = {ascent_dv + circ_dv:.1f} m/s ===")


if __name__ == "__main__":
    run_flight()