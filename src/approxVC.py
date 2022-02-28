import numpy as np
import time

def rand_index(n):
    # get random index
    return np.random.randint(0, n)


def approximateVC(E):
    C = []
    E_tag = E.copy()
    while len(E_tag) > 0:
        c = E_tag.pop(rand_index(len(E_tag)))
        C.append(c[0])
        C.append(c[1])
        for e in E_tag:
            if e[0] in C or e[1] in C:
                E_tag.remove(e)
    return len(C)


def get_edges(adj_matrix):
    # get edges from adjacency matrix
    edges = []
    for i in range(len(adj_matrix)):
        for j in range(i, len(adj_matrix)):
            if adj_matrix[i][j] == 1:
                edges.append((i, j))
    return edges


def main():
    start = time.time()
    data = np.loadtxt('../data/test_x.txt', dtype=int)
    matrix_shape = int(np.sqrt(len(data[0])))
    VC = []
    for x in data:
        matrix = x.reshape(matrix_shape, matrix_shape)
        VC.append(approximateVC(get_edges(matrix)))
        
    end = time.time()
    np.savetxt("../results/approxResults.txt", VC, fmt="%i")
    print(f"run took {end - start} seconds")

if __name__ == '__main__':
    main()