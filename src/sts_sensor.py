import numpy as np
class StsSensor:
    def __init__(self, address, position, orientation, size, resolution):
        self.address = address
        self.position = np.array(position, dtype=float)

        self.orientation = np.array(position, dtype=float)
        if len(orientation) == 2:
            self.theta = orientation[0]
            self.phi = orientation[1]

            Rz = np.array([
                [np.cos(self.phi), -np.sin(self.phi), 0],
                [np.sin(self.phi),  np.cos(self.phi), 0],
                [0, 0, 1]
            ])

            # Rotation around Y (theta)
            Ry = np.array([
                [np.cos(self.theta), 0, np.sin(self.theta)],
                [0, 1, 0],
                [-np.sin(self.theta), 0, np.cos(self.theta)]
            ])

            R = Rz @ Ry
            self.normal = np.array(R[:, 2], dtype=float)

        self.dx = 6
        self.dy = size
        self.dz = 0.03

        self.resolution = resolution
        self.res_x = self.resolution[0]
        self.res_y = self.resolution[1]
        self.res_z = self.resolution[2]

    def local_to_global(self, p_local):
        # Rotation matrices
        Rz = np.array([
            [np.cos(self.phi), -np.sin(self.phi), 0],
            [np.sin(self.phi),  np.cos(self.phi), 0],
            [0, 0, 1]
        ])
        Ry = np.array([
            [np.cos(self.theta), 0, np.sin(self.theta)],
            [0, 1, 0],
            [-np.sin(self.theta), 0, np.cos(self.theta)]
        ])
        R = Rz @ Ry
        T = np.array(self.position)
        return R @ p_local + T

    def global_to_local(self, p_global):
        Rz = np.array([
            [np.cos(self.phi), -np.sin(self.phi), 0],
            [np.sin(self.phi),  np.cos(self.phi), 0],
            [0, 0, 1]
        ])
        Ry = np.array([
            [np.cos(self.theta), 0, np.sin(self.theta)],
            [0, 1, 0],
            [-np.sin(self.theta), 0, np.cos(self.theta)]
        ])
        R = Rz @ Ry
        T = np.array(self.position)
        return R.T @ (p_global - T)

