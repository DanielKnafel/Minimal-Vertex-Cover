import math
import numpy as np

def make_adjs_from_mat(size, mat):
        edges = []
        for i in range(size):
            for j in range(i, size):
                if mat[i][j] == 1:
                    edges.append((i, j))
        return edges

def make_random_graph(size):
        mat = np.random.choice([0,1], size=(size, size),p=[0.8, 0.2])

        for i in range(size):
            mat[i][i] = 0
        mat = np.tril(mat)
        mat += mat.T
        return mat

class Graph:
    def __init__(self, size : int):
        self.size = size
        self.am = make_random_graph(size)
        self.E = make_adjs_from_mat(size, self.mat)
        self.V = range(size)

    def __init__(self, array : np.ndarray):
        size = int(math.sqrt(len(array)))
        self.size = size
        mat = array.reshape((size, size))
        self.E = make_adjs_from_mat(size, mat)
        self.V = range(size)
    
    @staticmethod
    def make_graphs_from_file(filename):
        graphs = []
        grap_vecs = np.loadtxt(filename)

        for vec in grap_vecs:
            graphs.append(Graph(vec))
        return graphs

    def is_connected(self, i, j):
        return bool(self.E[i][j])

    def print(self):
        print(self.am)
        print(self.E)