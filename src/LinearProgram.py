import time
from pulp import LpProblem, LpMinimize, LpVariable, LpBinary, lpSum, PULP_CBC_CMD, value
from Graph import Graph
import numpy as np

# open all graphs
graphs = Graph.make_graphs_from_file("../data/test_x.txt")
results = []

# start = time.time()
for i,G in enumerate(graphs):
    prob = LpProblem("MVC", LpMinimize)

    # define variables
    x = LpVariable.dicts("x", G.V, cat=LpBinary)
    y = LpVariable.dicts("y", G.E, cat=LpBinary)

    # objective function
    prob += lpSum(x)

    # constraints
    for (u,v) in G.E:
        prob += x[u] + x[v] >= 1

    # solve without printing
    prob.solve(PULP_CBC_CMD(msg=0))

    # print(f"number of vertices in solution : {prob.objective.value()}")
    results.append(int(prob.objective.value()))

    # display solution
    # for v in G.V:
    #     if value(x[v]) > 0.9:
    #         print(f"node {v} selected")

# end = time.time()
# print(f"run took {end-start} seconds")
np.savetxt("../results/lpResults.txt", results, fmt="%i")
