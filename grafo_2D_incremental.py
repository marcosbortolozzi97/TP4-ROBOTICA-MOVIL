#!/usr/bin/env python3
import gtsam
from lector_g2o import lector_g2o
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ===================================
# Cargamos datos 
# ===================================
poses, edges = lector_g2o("datasets/input_INTEL_g2o.g2o")

# =====================================================
# Creamos diccionario de poses por id para acceso rápido
# =====================================================
pose_dict = {p[0]: (p[1], p[2], p[3]) for p in poses}

# ===================================
# Configurar ISAM2
# ===================================
params = gtsam.ISAM2Params()
params.setRelinearizeThreshold(0.01)
params.relinearizeSkip = 10
isam = gtsam.ISAM2(params)

graph = gtsam.NonlinearFactorGraph()
initial = gtsam.Values()

# Prior inicial
prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.1]))
graph.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(0, 0, 0), prior_noise))
initial.insert(0, gtsam.Pose2(0, 0, 0))
inserted_keys = set([0])

# ===================================
# Procesamiento incremental
# ===================================
for idx, edge in enumerate(edges):
    i, j, dx, dy, dtheta, info = edge

    Omega = np.array([
        [info[0], info[1], info[2]],
        [info[1], info[3], info[4]],
        [info[2], info[4], info[5]]
    ])
    try:
        covariance = np.linalg.inv(Omega)
    except np.linalg.LinAlgError:
        Omega += np.eye(3) * 1e-6
        covariance = np.linalg.inv(Omega)

    model = gtsam.noiseModel.Gaussian.Covariance(covariance)
    relative_pose = gtsam.Pose2(dx, dy, dtheta)
    graph.add(gtsam.BetweenFactorPose2(i, j, relative_pose, model))

    if j not in inserted_keys:
        if j in pose_dict:
            xj, yj, thj = pose_dict[j]
            initial.insert(j, gtsam.Pose2(xj, yj, thj))
        else:
            initial.insert(j, gtsam.Pose2(0, 0, 0))
        inserted_keys.add(j)

    try:
        isam.update(graph, initial)
        graph.resize(0)
        initial.clear()
    except Exception as e:
        print(f"Warning: isam.update falló en iter {idx} (edge {i}->{j}): {e}")
        graph.resize(0)
        initial.clear()

# ===================================
# Resultado final 
# ===================================
result = isam.calculateEstimate()

# ===================================
# Funciones auxiliares
# ===================================
def trajectory_from_values(values):
    xs, ys = [], []
    keys = []
    try:
        keys = list(values.keys())
    except Exception:
        for k in range(values.size()):
            try:
                _ = values.atPose2(k)
                keys.append(k)
            except Exception:
                continue
    for k in sorted(keys):
        try:
            pose = values.atPose2(k)
            xs.append(pose.x())
            ys.append(pose.y())
        except Exception:
            continue
    return xs, ys

# ===================================
# Visualización y Guardado
# ===================================
# Trayectoria inicial
x0 = [p[1] for p in poses]
y0 = [p[2] for p in poses]

# Trayectoria optimizada
xs, ys = trajectory_from_values(result)

plt.figure(figsize=(10,8))
plt.plot(x0, y0, 'r--', linewidth=1.2, label='Inicial (sin optimizar)')
plt.plot(xs, ys, 'b-', linewidth=1.5, label='Optimizada (iSAM2)')
plt.scatter(xs[0], ys[0], c='green', label='Inicio')
plt.title("Comparación de Trayectorias - Graph SLAM Incremental (iSAM2)")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.savefig("2D_isam.png")
print("Se guarda la imagen como 2D_isam.png")

