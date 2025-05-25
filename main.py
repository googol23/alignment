import sys
sys.path.append('src')
import time
import numpy as np
import matplotlib.pyplot as plt

from boost_histogram import Histogram
from boost_histogram.axis import Regular

from sts_sensor import StsSensor
from sts_track import StsTrack
from particle_gun import ParticleGun
from trajectories import *
from fitters import fit_least_squares
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# Multiprocessing setup
def split_into_batches(data, n_batches):
    avg = len(data) // n_batches
    return [data[i * avg : (i + 1) * avg] for i in range(n_batches - 1)] + [data[(n_batches - 1) * avg:]]
n_of_threads = 10

# Particle source
gun = ParticleGun(r0=[.0, .0, .0], dr=[.0, .0, 0], p0=[0, 0, 1], dp=[.03, .03, .0], mass=0.938)

# Geometry
sensors = [StsSensor(i,(0,0,10*i), 6, 0, 0, (0.1, 0.01, 0.0)) for i in range(8)]


def process_particles(particles, worker_id=0):
    h_res_rho_local = [Histogram(Regular(100, -0.5, +0.5)) for _ in sensors]
    trajectories = []
    measurements = []
    fit_results = []

    # for p in particles:
    for p in tqdm(particles, position=worker_id, desc=f"Worker {worker_id}", leave=False):
        # trajectory = real_trajectory(p, sensors)
        # trajectories.append(trajectory)

        measurement = measured_trajectory(p, sensors)
        measurements.append(measurement)

    return trajectories, measurements

def print_measurement(measurement):
    str_msr = ""
    for hit in msr:
        if hit is not None:
            str_msr += f"{hit[0]:+.3f},{hit[1]:+.3f},{hit[2]:+.3f},"
        else:
            str_msr += f"None,None,None,"
    return str_msr

if __name__ == "__main__":
    # Step 1: Generate particles
    start = time.perf_counter()
    particles = gun.generate(1000)
    end = time.perf_counter()
    print(f"Particle generation took {end - start:.3f} seconds")

    # Step 2: Split into batches
    start = time.perf_counter()
    batches = split_into_batches(particles, n_of_threads)
    end = time.perf_counter()
    print(f"Splitting into batches took {end - start:.3f} seconds")

    # Step 3: Parallel processing
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_of_threads) as executor:
        futures = [
            executor.submit(process_particles, batch, i)
            for i, batch in enumerate(batches)
        ]
        with open(f"measurements.csv", "w") as f:
            for future in futures:
                try:
                    trajectories, measurements = future.result()

                    for msr in measurements:
                        f.write(f"{print_measurement(msr)}\n")
                except Exception as e:
                    print(f"Error processing particle: {e}")
    end = time.perf_counter()
    print(f"Parallel processing took {end - start:.3f} seconds")
