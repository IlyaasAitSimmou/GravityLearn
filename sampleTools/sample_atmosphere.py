import krpc
import numpy as np
import json
import math

conn = krpc.connect(rpc_port=50010, stream_port=50011)
vessel = conn.space_center.active_vessel
body = vessel.orbit.body
ref = body.non_rotating_reference_frame
radius = body.equatorial_radius

direction = np.array([1.0, 0.0, 0.0])
altitudes = list(range(0, 70001, 500))
densities = []
pressures = []
speeds_of_sound = []

for alt in altitudes:
    r = direction * (radius + alt)
    rho = body.atmospheric_density_at_position(tuple(r), ref)
    pressure = body.pressure_at(float(alt))
    temp = body.temperature_at(tuple(r), ref)
    # Ideal gas law for speed of sound on Kerbin (gamma=1.4, R=287.058)
    sos = math.sqrt(1.4 * 287.058 * max(temp, 1.0))

    densities.append(rho)
    pressures.append(pressure)
    speeds_of_sound.append(sos)
    print(f"alt={alt:>6}m  rho={rho:.6f} kg/m³  p={pressure:.1f} Pa  sos={sos:.1f} m/s")

with open("kerbin_atmosphere_table.json", "w") as f:
    json.dump({
        "altitudes": altitudes,
        "densities": densities,
        "pressures": pressures,
        "speeds_of_sound": speeds_of_sound
    }, f, indent=2)
print(f"\nSaved {len(altitudes)} samples.")
