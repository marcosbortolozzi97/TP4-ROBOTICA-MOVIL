import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gtsam
from gtsam import Pose3, Rot3, Point3
from lector_g2o_3D import read_g2o_3d

def normalize_quaternion(qw, qx, qy, qz):
    q = np.array([qw, qx, qy, qz], dtype=float)
    return q / np.linalg.norm(q)

def info_vector_to_matrix6x6(info_vec):
    I = np.zeros((6, 6))
    idx = 0
    for r in range(6):
        for c in range(r, 6):
            I[r, c] = info_vec[idx]
            I[c, r] = info_vec[idx]
            idx += 1
    return I

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max([x_range, y_range, z_range])
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
    ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
    ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

# ================================================================
# Cargamos el archivo .g2o
# ================================================================
file_path = "datasets/parking-garage.g2o"
poses, edges = read_g2o_3d(file_path)

print(f"Total de poses: {len(poses)}")
print(f"Total de aristas: {len(edges)}")

# ================================================================
#  Parámetros ISAM2
# ================================================================
isam_params = gtsam.ISAM2Params()
isam_params.setRelinearizeThreshold(0.01)
isam_params.relinearizeSkip = 10 
isam = gtsam.ISAM2(isam_params)

graph = gtsam.NonlinearFactorGraph()
initial = gtsam.Values()

# Prior usando primera pose real
_, x0, y0, z0, qx0, qy0, qz0, qw0 = poses[0]
qw0, qx0, qy0, qz0 = normalize_quaternion(qw0, qx0, qy0, qz0)
rot0 = Rot3.Quaternion(qw0, qx0, qy0, qz0)
trans0 = Point3(x0, y0, z0)
prior_pose = Pose3(rot0, trans0)

prior_noise = gtsam.noiseModel.Diagonal.Variances(np.array([1e-6]*6))
graph.add(gtsam.PriorFactorPose3(0, prior_pose, prior_noise))
initial.insert(0, prior_pose)

# Insertamos el vector de poses iniciales 
x_init = np.array([p[1] for p in poses])
y_init = np.array([p[2] for p in poses])
z_init = np.array([p[3] for p in poses])

# ================================================================
# Procesamiento incremental
# ================================================================
inserted_keys = set([0])   # claves ya incorporadas dentro de isam (se asigna la 0 como prior)
to_insert = set()          # claves que vamos a insertar en la próxima llamada a isam.update()

# ================================================================
# recorrida de aristas
# ================================================================
for i, edge in enumerate(edges):
    idx_i = int(edge[0])
    idx_j = int(edge[1])
    dx, dy, dz = edge[2], edge[3], edge[4]

    # normalizamos cuaternión del edge (orden del g2o: qx,qy,qz,qw)
    qx, qy, qz, qw = edge[5], edge[6], edge[7], edge[8]
    qw, qx, qy, qz = normalize_quaternion(qw, qx, qy, qz)

    # construimos factor relativo
    rot_rel = Rot3.Quaternion(qw, qx, qy, qz)
    trans_rel = Point3(dx, dy, dz)
    rel_pose = Pose3(rot_rel, trans_rel)

    # reconstruimos matriz info a cov
    info_vec = edge[9]
    info_mat = info_vector_to_matrix6x6(info_vec)
    info_mat += np.eye(6) * 1e-8
    cov = np.linalg.inv(info_mat)
    noise = gtsam.noiseModel.Gaussian.Covariance(cov)

    # agregamos factor temporal
    graph.add(gtsam.BetweenFactorPose3(idx_i, idx_j, rel_pose, noise))

    # si necesitamos una estimación inicial para idx_j, las insertamos solo si no existe
    if (idx_j not in inserted_keys) and (idx_j not in to_insert):
        # intentamos obtener la pose desde 'poses' y preparamos initial
        try:
            p = next(p for p in poses if p[0] == idx_j)
            _, xj, yj, zj, qxj, qyj, qzj, qwj = p
            qw_j, qx_j, qy_j, qz_j = normalize_quaternion(qwj, qxj, qyj, qzj)
            rot_j = Rot3.Quaternion(qw_j, qx_j, qy_j, qz_j)
            trans_j = Point3(xj, yj, zj)
            initial.insert(idx_j, Pose3(rot_j, trans_j))
        except StopIteration:
            # fallback: si existe idx_i en initial o en isam, se compone
            if initial.exists(idx_i):
                initial.insert(idx_j, initial.atPose3(idx_i).compose(rel_pose))
            elif idx_i in inserted_keys:
                # si idx_i ya está en isam, componemos desde la estimación de isam
                try:
                    est_i = isam.calculateEstimate().atPose3(idx_i)
                    initial.insert(idx_j, est_i.compose(rel_pose))
                except Exception:
                    initial.insert(idx_j, Pose3())
            else:
                initial.insert(idx_j, Pose3())

        # marcar que idx_j está programada para ser agregada a isam en la próxima carga
        to_insert.add(idx_j)

    # condición de actualización: se usa una actualización periódica cada 50 factores 
    if i % 50 == 0:
        # si hay algo que pasar, actualizamos
        if (graph.size() > 0) or (initial.size() > 0):
            isam.update(graph, initial)
            # mover claves programadas a las ya insertadas
            inserted_keys.update(to_insert)
            to_insert.clear()
            graph.resize(0)
            initial.clear()

# ================================================================
# última actualización 
# ================================================================
if (graph.size() > 0) or (initial.size() > 0):
    isam.update(graph, initial)
    inserted_keys.update(to_insert)
    to_insert.clear()
    graph.resize(0)
    initial.clear()

# ================================================================
# calculamos la estimacion final 
# ================================================================
result = isam.calculateEstimate()

# ================================================================
# Extraemos la trayectoria optimizada
# ================================================================
x_opt, y_opt, z_opt = [], [], []
for k in range(len(poses)):
    try:
        p = result.atPose3(k)
        
        try:
            tx = p.x(); ty = p.y(); tz = p.z()
        except Exception:
            t = p.translation()
            tx, ty, tz = float(t[0]), float(t[1]), float(t[2])
        x_opt.append(tx); y_opt.append(ty); z_opt.append(tz)
    except Exception:
        # si falta alguna clave, agregamos nan para mantener índices
        x_opt.append(np.nan); y_opt.append(np.nan); z_opt.append(np.nan)

x_opt = np.array(x_opt); y_opt = np.array(y_opt); z_opt = np.array(z_opt)

# ================================================================
# Filtramos NaNs al graficar
# ================================================================
valid = ~np.isnan(x_opt)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_init, y_init, z_init, 'r--', label="Inicial (g2o)")
ax.plot(x_opt[valid], y_opt[valid], z_opt[valid], 'b-', label="Optimizada")
ax.scatter([x_init[0]], [y_init[0]], [z_init[0]], c='g', s=80, label="Inicio")
ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
ax.set_title("Graph-SLAM 3D - iSAM2")
set_axes_equal(ax)
ax.legend()
plt.tight_layout()
plt.savefig("3D_isam.png")
print("Guardado como 3D_isam.png")
