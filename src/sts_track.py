import numpy as np

class StsTrack:
    def __init__(self, r0, p, mass):
        self.r0 = r0
        self.x = r0[0]
        self.y = r0[1]
        self.z = r0[2]
        self.tx = p[0] / p[2]
        self.ty = p[1] / p[2]
        self.tz = 1
        self.T = [self.tx, self.ty, self.tz]
        self.mass = mass

        p_mag = np.linalg.norm(p)
        gamma = np.sqrt(1 + (p_mag / (self.mass))**2)
        self.velocity = p / (gamma * self.mass)

    def __str__(self):
        return f"Track: x={self.x:+.2f}, y={self.y:+.2f}, z={self.z:+.2f}, tx={self.tx:+.2f}, ty={self.ty:+.2f}, tz={self.tz:+.2f}"