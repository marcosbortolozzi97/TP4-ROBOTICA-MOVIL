def read_g2o_3d(filename):
    """
    Lee un archivo .g2o con datos 3D (VERTEX_SE3:QUAT y EDGE_SE3:QUAT)
    Devuelve listas de poses y aristas.
    """
    poses = []
    edges = []

    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue

            tag = parts[0]
            if tag == "VERTEX_SE3:QUAT":
                # Formato: VERTEX_SE3:QUAT i x y z qx qy qz qw
                idx = int(parts[1])
                x, y, z = map(float, parts[2:5])
                qx, qy, qz, qw = map(float, parts[5:9])
                poses.append([idx, x, y, z, qx, qy, qz, qw])

            elif tag == "EDGE_SE3:QUAT":
                # Formato: EDGE_SE3:QUAT i j x y z qx qy qz qw info[21]
                i, j = int(parts[1]), int(parts[2])
                x, y, z = map(float, parts[3:6])
                qx, qy, qz, qw = map(float, parts[6:10])
                info = list(map(float, parts[10:31]))  # 21 valores
                edges.append([i, j, x, y, z, qx, qy, qz, qw, info])

    return poses, edges


if __name__ == "__main__":
    filename = "datasets/parking-garage.g2o"
    poses, edges = read_g2o_3d(filename)
    print(f"Total de poses: {len(poses)}")
    print(f"Total de aristas: {len(edges)}")
    print("Primera pose:", poses[0])
    print("Primera arista:", edges[0])
