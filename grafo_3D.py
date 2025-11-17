#!/usr/bin/env python3
import gtsam
from lector_g2o_3D import read_g2o_3d
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================================================================
# Convertimos el vector de 21 elementos a matriz 6x6
# ================================================================
def info_vector_to_matrix6x6(info):
    I = np.zeros((6, 6))
    upper = np.triu_indices(6)
    I[upper] = info
    I = I + np.triu(I, 1).T
    return I

# ================================================================
# Cargamos el dataset
# ================================================================
poses, edges = read_g2o_3d("datasets/parking-garage.g2o")

print(f"Total de poses: {len(poses)}")
print(f"Total de aristas: {len(edges)}")

# ================================================================
# Creamos el grafo de factores
# ================================================================
graph = gtsam.NonlinearFactorGraph()

prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
    np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1])
)
graph.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(), prior_noise))

# ================================================================
# Agregamos los factores BetweenFactorPose3
# ================================================================
for edge in edges:
    i, j, dx, dy, dz, qx, qy, qz, qw, info = edge

    # Matriz de información → covarianza
    info_matrix = info_vector_to_matrix6x6(info)
    try:
        covariance = np.linalg.inv(info_matrix)
    except np.linalg.LinAlgError:
        info_matrix += np.eye(6) * 1e-6
        covariance = np.linalg.inv(info_matrix)

    model = gtsam.noiseModel.Gaussian.Covariance(covariance)

    rot_rel = gtsam.Rot3.Quaternion(qw, qx, qy, qz)
    trans_rel = gtsam.Point3(dx, dy, dz)
    rel_pose = gtsam.Pose3(rot_rel, trans_rel)

    graph.add(gtsam.BetweenFactorPose3(i, j, rel_pose, model))

# ================================================================
# Construccimos la estimación inicial (como en iSAM2)
# ================================================================
initial = gtsam.Values()
current_estimates = {}

# Pose inicial (0)
idx0, x0, y0, z0, qx0, qy0, qz0, qw0 = poses[0]
rot0 = gtsam.Rot3.Quaternion(qw0, qx0, qy0, qz0)
p0 = gtsam.Pose3(rot0, gtsam.Point3(x0, y0, z0))

initial.insert(0, p0)
current_estimates[0] = p0

# Se propaga usando odometría
for edge in edges:
    i, j, dx, dy, dz, qx, qy, qz, qw, info = edge

    if i in current_estimates and j not in current_estimates:
        rot_rel = gtsam.Rot3.Quaternion(qw, qx, qy, qz)
        trans_rel = gtsam.Point3(dx, dy, dz)
        rel_pose = gtsam.Pose3(rot_rel, trans_rel)

        current_estimates[j] = current_estimates[i].compose(rel_pose)

# Insertar al Values final
for k, p in current_estimates.items():
    if not initial.exists(k):
        initial.insert(k, p)


print(f"Estimación inicial construida para {len(current_estimates)} poses.")

# ================================================================
# Optimización (Gauss–Newton)
# ================================================================
params = gtsam.GaussNewtonParams()
params.setVerbosity("ERROR")

optimizer = gtsam.GaussNewtonOptimizer(graph, initial, params)
result = optimizer.optimize()

print("Optimización completada con éxito")

# ================================================================
# Extraemos las trayectorias
# ================================================================
def extract_trajectory(values):
    xs, ys, zs = [], [], []
    for k in sorted(values.keys()):
        pose = values.atPose3(k)
        t = pose.translation()
        xs.append(t[0])
        ys.append(t[1])
        zs.append(t[2])
    return np.array(xs), np.array(ys), np.array(zs)

x0, y0, z0 = extract_trajectory(initial)
x1, y1, z1 = extract_trajectory(result)

# ================================================================
# Visualización
# ================================================================

def set_axes_equal(ax):
    """Hace que la escala de los ejes X, Y, Z sea igual para evitar distorsiones."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max([x_range, y_range, z_range])

    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    ax.set_xlim3d([x_mid - max_range/2, x_mid + max_range/2])
    ax.set_ylim3d([y_mid - max_range/2, y_mid + max_range/2])
    ax.set_zlim3d([z_mid - max_range/2, z_mid + max_range/2])


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

ax.plot(x0, y0, z0, 'r--', linewidth=1.0, label="Trayectoria inicial")
ax.plot(x1, y1, z1, 'b-', linewidth=1.5, label="Trayectoria optimizada")

ax.scatter(x1[0], y1[0], z1[0], c='green', s=30, label="Inicio")

ax.set_title("Graph-SLAM 3D (Gauss–Newton)")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.legend()
ax.grid(True)
¡
set_axes_equal(ax)

plt.tight_layout()
plt.savefig("3D_Batch.png")

print("Guardado como 3D_Batch.png")

