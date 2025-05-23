class StsSensor:
    def __init__(self, position, size, theta, phi, resolution):
        self.position = position
        self.x = position[0]
        self.y = position[1]
        self.z = position[2]
        self.theta = theta
        self.phi = phi

        self.dx = 6
        self.dy = size
        self.dz = 0.03

        self.resolution = resolution
        self.res_x = self.resolution[0]
        self.res_y = self.resolution[1]
        self.res_z = self.resolution[2]