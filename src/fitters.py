import numpy as np

from scipy.optimize import least_squares
def fit_least_squares(points):
    points = np.asarray(points)
    z = points[:, 2]
    x = points[:, 0]
    y = points[:, 1]

    # Residual function: distances from points to the line parameterized by tx, x0 and ty, y0
    def residuals(params):
        tx, x0, ty, y0 = params
        x_fit = tx * z + x0
        y_fit = ty * z + y0
        return np.sqrt((x - x_fit)**2 + (y - y_fit)**2)

    # Initial guess
    p0 = [0, np.mean(x), 0, np.mean(y)]

    res = least_squares(residuals, p0)

    tx, x0, ty, y0 = res.x
    z0 = np.mean(z)
    r0 = np.array([tx*z0 + x0, ty*z0 + y0, z0])

    # Caclulate residuals
    v = np.array([tx, ty, 1.0])
    v /= np.linalg.norm(v)
    diffs = np.asarray(points) - r0
    proj = np.outer(np.dot(diffs, v), v)
    perp = diffs - proj

    return r0, tx, ty, np.linalg.norm(perp, axis=1)

from filterpy.kalman import KalmanFilter
def fit_kf(points):
    points = np.asarray(points)
    n = len(points)
    if n < 2:
        raise ValueError("Need at least two points to fit a line")

    # State: [x, y, z, vx, vy, vz]
    kf = KalmanFilter(dim_x=6, dim_z=3)
    dt = 1.0  # time step (arbitrary units, consistent for all points)

    # State transition matrix (constant velocity)
    kf.F = np.array([
        [1, 0, 0, dt, 0,  0],
        [0, 1, 0, 0,  dt, 0],
        [0, 0, 1, 0,  0,  dt],
        [0, 0, 0, 1,  0,  0],
        [0, 0, 0, 0,  1,  0],
        [0, 0, 0, 0,  0,  1],
    ])

    # Measurement function: we measure position only
    kf.H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
    ])

    kf.P *= 1e3  # Initial covariance
    kf.R *= 0.01  # Measurement noise (adjust if needed)
    kf.Q = np.eye(6) * 1e-5  # Process noise

    # Initialize state with first point, zero velocity
    kf.x[:3, 0] = points[0]
    kf.x[3:, 0] = 0.

    # We simulate time steps as increments of 1 between points
    for i in range(n):
        kf.predict()
        kf.update(points[i])

    # Extract fitted line parameters from final state
    r0 = kf.x[:3, 0]        # position vector
    v = kf.x[3:, 0]         # velocity vector

    # Normalize direction so vz = 1
    if abs(v[2]) < 1e-8:
        raise ValueError("Velocity z-component too small to define line slopes")

    tx = v[0] / v[2]
    ty = v[1] / v[2]

    # Caclulate residuals
    v = np.array([tx, ty, 1.0])
    v /= np.linalg.norm(v)
    diffs = np.asarray(points) - r0
    proj = np.outer(np.dot(diffs, v), v)
    perp = diffs - proj

    return r0, tx, ty, np.linalg.norm(perp, axis=1)

def line_model(B, z):
    return B[0]*z + B[1]

from scipy.odr import ODR, Model, RealData
def fit_odr(points):
    points = np.asarray(points)
    z = points[:, 2]
    x = points[:, 0]
    y = points[:, 1]

    model = Model(line_model)
    tx, x0 = ODR(RealData(z, x), model, beta0=[0., 0.]).run().beta
    ty, y0 = ODR(RealData(z, y), model, beta0=[0., 0.]).run().beta

    z0 = np.mean(z)
    r0 = np.array([tx*z0 + x0, ty*z0 + y0, z0])

    # Caclulate residuals
    v = np.array([tx, ty, 1.0])
    v /= np.linalg.norm(v)
    diffs = np.asarray(points) - r0
    proj = np.outer(np.dot(diffs, v), v)
    perp = diffs - proj

    return r0, tx, ty, np.linalg.norm(perp, axis=1)

def fit_pca(points):
    points = np.asarray(points)
    centroid = points.mean(axis=0)

    # Subtract centroid
    pts_centered = points - centroid

    # Covariance matrix
    cov = np.cov(pts_centered.T)

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Direction vector: eigenvector with max eigenvalue
    direction = eigvecs[:, np.argmax(eigvals)]

    # Normalize direction
    direction /= np.linalg.norm(direction)

    # Slopes tx, ty relative to z (if direction_z != 0)
    if abs(direction[2]) < 1e-8:
        raise ValueError("Direction vector is almost perpendicular to z-axis; can't define slopes tx, ty reliably.")

    scale = 1.0 / direction[2]
    direction = direction * scale  # (tx, ty, 1)

    tx = direction[0]
    ty = direction[1]

    # Caclulate residuals
    diff = points - centroid  # shape (N,3)
    proj_lengths = np.dot(diff, direction)  # shape (N,)
    proj_vectors = np.outer(proj_lengths, direction)  # shape (N,3)
    perp_vectors = diff - proj_vectors
    residuals = np.linalg.norm(perp_vectors, axis=1)

    return centroid, tx, ty, residuals