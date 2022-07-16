import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import linalg


# get Top Down Left Right points
def get_neighbours(indice):
    i, j = indice
    return [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]


# subtraction of value at point i and its four neighbours
def f(src, indice):
    i, j = indice
    neighbour_sum = int(src[i-1, j]) + int(src[i+1, j]) + int(src[i, j-1]) + int(src[i, j+1])
    return 4 * int(src[i, j]) - neighbour_sum


def construct_sparse_matrix(indices):
    n = len(indices)      # number of nonzero points in mask matrix
    # A = csr_matrix((n, n))
    A = lil_matrix((n, n))

    for i, indice in enumerate(indices):
        A[i, i] = 4
        # set value at (x,y) which is inside of source image to -1
        for neighbour in get_neighbours(indice):
            if neighbour in indices:
                j = indices.index(neighbour)
                A[i, j] = -1
    return A


# if yes, return those neighbours' indices; if no, return []
def get_outside_neighbours(mask, x):
    js = []
    # if not mask[x]:
    #     return js
    for neighbour in get_neighbours(x):
        if not mask[neighbour]:  # outside boundary
            js.append(neighbour)
    return js


def blend(source, target, mask):
    # get nonzero values' indices of mask. e.g.(0,1),(1,3)
    nonzero = np.nonzero(mask)
    indices = list(zip(nonzero[0], nonzero[1]))

    n = len(indices)

    # construct sparse matrix A
    A = construct_sparse_matrix(indices)
    # initialize matrix b
    b = np.zeros(n)

    # handle points outside of boundary but are neighbours to points inside of boundary
    for (i, indice) in enumerate(indices):
        b[i] = f(source, indice)    # 4*si - (sj1+sj2+sj3+sj4)
        # if i has neighbour j outside of boundary, plus target[j] to b[i]
        js = get_outside_neighbours(mask, indice)
        if len(js) != 0:
            for j in js:
                b[i] += target[j]

    # solve for x
    x = linalg.cg(A, b)
    # x = spsolve(A, b)

    # copy target image
    # when change the data type to uint8, get weird result
    target_copy = np.copy(target).astype(int)
    # replace intensities
    for i, indice in enumerate(indices):
        target_copy[indice] = x[0][i]
    # print(target_copy)
    return target_copy
