import numpy as np

from sts_track import StsTrack

class ParticleGun:
    def __init__(self, r0:list = [0,0,0], dr:list = [0,0,0], p0:list = [0,0,0], dp:list = [0,0,0], mass:float = 0.938):
        self.r0 = r0
        self.dr = dr
        self.p0 = p0
        self.dp = dp
        self.mass = mass

    def generate(self, n_particles = 1):
        particles = []
        for i in range(n_particles):
            r0 = [np.random.normal(xi, self.dr[i]) for i,xi in enumerate(self.r0)]
            p  = [np.random.normal(pi, self.dp[i]) for i,pi in enumerate(self.p0)]
            particles.append(StsTrack(r0, p, self.mass))
        return particles