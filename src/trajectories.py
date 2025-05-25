import numpy as np
from sts_track import StsTrack

def real_trajectory(track: StsTrack, sensors: list, n_points=100):
    z = np.linspace(sensors[0].z, sensors[-1].z, n_points)
    x = track.x + track.tx * (z - track.x)
    y = track.y + track.ty * (z - track.y)
    return x, y, z

def measured_trajectory(track: StsTrack, sensors: list):
    measurements = []
    for sensor in sensors:
        # Calculate intersection with the sensor plane
        if track.tz == 0:
            continue  # Avoid division by zero

        x_intersect = track.x + track.tx * (sensor.z - track.x)
        y_intersect = track.y + track.ty * (sensor.z - track.y)

        # Smear measurements by sensor resolution
        x_intersect += np.random.normal(0, sensor.res_x)
        y_intersect += np.random.normal(0, sensor.res_y)

        # Check if within sensor bounds
        if (sensor.x - sensor.dx/2 <= x_intersect <= sensor.x + sensor.dx/2 and
            sensor.y - sensor.dy/2 <= y_intersect <= sensor.y + sensor.dy/2):
            measurements.append((sensor.address, x_intersect, y_intersect, sensor.z))

    return measurements
