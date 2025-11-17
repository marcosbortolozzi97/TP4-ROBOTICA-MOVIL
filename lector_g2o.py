def lector_g2o(filename):
    """
    Lee un archivo 2D .g2o y devuelve listas de poses y aristas.
    """
    poses = []
    edges = []

    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue

            tag = parts[0]
            if tag == "VERTEX_SE2":
                # Ejemplo: VERTEX_SE2 id x y theta
                idx = int(parts[1])
                x, y, theta = map(float, parts[2:5])
                poses.append([idx, x, y, theta])

            elif tag == "EDGE_SE2":
                # Ejemplo: EDGE_SE2 i j dx dy dtheta info[6]
                i, j = int(parts[1]), int(parts[2])
                dx, dy, dtheta = map(float, parts[3:6])
                info = list(map(float, parts[6:]))
                edges.append([i, j, dx, dy, dtheta, info])

    return poses, edges


if __name__ == "__main__":
    # Ruta al dataset
    filename = "datasets/input_INTEL_g2o.g2o"
    poses, edges = lector_g2o(filename)
    print(f"Total de poses: {len(poses)}")
    print(f"Total de aristas: {len(edges)}")
    print("Primera pose:", poses[0])
    print("Primera arista:", edges[0])
