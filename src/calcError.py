import numpy as np
import sys

def find_error(list1, list2):
    error = 0
    for i in range(len(list1)):
        error += np.abs(list1[i] - list2[i])
    return error/len(list1)


def main():
    true_results = '../data/test_y.txt'
    test_results  = sys.argv[1]
    list1 = np.loadtxt(test_results, dtype=int)
    list2 = np.loadtxt(true_results, dtype=int)
    print(find_error(list1, list2))


if __name__ == '__main__':
    main()