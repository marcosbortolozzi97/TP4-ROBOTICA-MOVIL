import gtsam
from lector_g2o import lector_g2o
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Cargamos datos del archivo .g2o
# ==========================
poses, edges = lector_g2o("datasets/input_INTEL_g2o.g2o")

# ==========================
# Creamos grafo de factores
# ==========================
graph = gtsam.NonlinearFactorGraph()

# Ruido del prior (pose inicial)
prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.1]))  # [x, y, theta]
graph.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(0, 0, 0), prior_noise))

# ==========================
# Agregamos factores (edges)
# ==========================
for edge in edges:
    i, j, dx, dy, dtheta, info = edge
    # Construimos la matriz de información y su covarianza
    Omega = np.array([
        [info[0], info[1], info[2]],
        [info[1], info[3], info[4]],
        [info[2], info[4], info[5]]
    ])
    covariance = np.linalg.inv(Omega)
    model = gtsam.noiseModel.Gaussian.Covariance(covariance)
    relative_pose = gtsam.Pose2(dx, dy, dtheta)
    graph.add(gtsam.BetweenFactorPose2(i, j, relative_pose, model))

# ==========================
# Estimación inicial
# ==========================
initial = gtsam.Values()
for pose in poses:
    idx, x, y, theta = pose
    initial.insert(idx, gtsam.Pose2(x, y, theta))

# ==========================
# Optimización (Gauss-Newton)
# ==========================
params = gtsam.GaussNewtonParams()
params.setVerbosity("ERROR")  
optimizer = gtsam.GaussNewtonOptimizer(graph, initial, params)
result = optimizer.optimize()

# ==========================
# Visualización
# ==========================
def plot_trajectory(values, color, label):
    xs, ys = [], []
    for k in range(values.size()):
        pose = values.atPose2(k)
        xs.append(pose.x())
        ys.append(pose.y())
    plt.plot(xs, ys, color=color, label=label, linewidth=1.5)

plt.figure(figsize=(8,6))
plot_trajectory(initial, 'r', 'Inicial')
plot_trajectory(result, 'b', 'Optimizada')
plt.legend()
plt.title("Optimización Graph-SLAM 2D (Gauss-Newton)")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.grid(True)

plt.savefig("2D_Batch.png", dpi=300)
print("Imagen guardada como 2D_Batch.png")
