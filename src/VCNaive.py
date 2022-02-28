import itertools
import numpy as np
from Graph import Graph
import time

def get_sub_graphs(G, k):
    r = range(0, G.size)
    combs = list(itertools.combinations(r, k))
    return combs

def is_VC(G, comb):
    edges_touched = 0
    
    for e in G.E:
        if e[0] in comb or e[1] in comb:
            edges_touched += 1

    return True if edges_touched == len(G.E) else False
    
def find_min_VC(G):
    for k in range (1, G.size):
        combs = get_sub_graphs(G, k)
        for comb in combs:
            if is_VC(G, comb):
                return comb
    return None

def make_dataset(dataset_size, train_x_fname, train_y_fname):
    graph_size = 15
    with open(train_x_fname,'w') as train_x:
        with open(train_y_fname,'w') as train_y:
            for i in range(dataset_size):
                if i % 1000 == 0:
                    print(i)
                g = Graph(graph_size)
                vc = find_min_VC(g)
                np.savetxt(train_x, g.am.reshape((1, -1)), fmt="%i")
                train_y.write(str(len(vc)) + "\n")

start = time.time()
graphs = Graph.make_graphs_from_file("../data/test_x.txt")
results = [len(find_min_VC(g)) for g in graphs]
end = time.time()
print(f"run took {end-start} seconds")
np.savetxt("../results/naiveResults.txt", results, fmt="%i")
