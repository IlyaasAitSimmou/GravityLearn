import time

class VerticalAscent():
    def __init__(self, vessel, conn):
        self.vessel = vessel
        self.conn = conn
        self.surface_altitude = conn.add_stream(
            getattr, vessel.flight(), 'surface_altitude'
        )
    
    def run(self, target_altitude):
        print("Phase: Vertical ascent")
        while self.surface_altitude() < target_altitude:
            self.check_staging()
            time.sleep(0.1)

        print(f"Altitude {target_altitude}m reached, commencing pitch program")
    
    def check_staging(self):
        # First-stage-only contract: never auto-stage. has_fuel briefly reads
        # False in the first frames after ignition, which used to fire the
        # next stage here and detach the first stage on multi-stage rockets.
        pass
