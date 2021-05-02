import numpy

try:
    import cupy
except ImportError:
    cupy = None


def supports_cupy():
    return cupy is not None


def get_cupy():
    return cupy


def get_array_module(x):
    if cupy is not None:
        return cupy.get_array_module(x)
    else:
        return numpy


def asnumpy(x):
    if cupy is not None:
        return cupy.asnumpy(x)
    else:
        return numpy.asarray(x)


def find_nearest_neighbor(src_matrix, trg_matrix):
    xp = get_array_module(src_matrix)
    matrix_mult = xp.dot(src_matrix, trg_matrix.transpose())
    neighbors = xp.argmax(matrix_mult, axis=1).tolist()
    neighbors = [neighbor[0] for neighbor in neighbors]
    return neighbors


def normalize_matrix(matrix):
    xp = get_array_module(matrix)
    norms = xp.sqrt(xp.sum(xp.square(matrix), axis=1))
    norms[norms == 0] = 1
    norms = norms.reshape(-1, 1)
    matrix /= norms[:]
    return matrix


def check_if_neighbors_match(src_neighbor, trg_neighbor):
    matching = {}
    for current_neighbor_index in range(len(src_neighbor)):
        # Looking for matches
        if int(trg_neighbor[src_neighbor[current_neighbor_index]]) == current_neighbor_index:
            matching[current_neighbor_index] = src_neighbor[current_neighbor_index]

    return matching


def mean_center(matrix):
    xp = get_array_module(matrix)
    avg = xp.mean(matrix, axis=0)
    matrix -= avg
    return matrix


def vecmap_normalize(matrix):
    matrix = normalize_matrix(matrix)
    matrix = mean_center(matrix)
    matrix = normalize_matrix(matrix)

    return matrix


def topk_mean(m, k, inplace=False):  # TODO Assuming that axis is 1
    xp = get_array_module(m)
    n = m.shape[0]
    ans = xp.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = xp.array(m)
    ind0 = xp.arange(n)
    ind1 = xp.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        m.argmax(axis=1, out=ind1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    return ans / k


def dropout(m, p):
    xp = get_array_module(m)
    if p <= 0.0:
        return m
    else:
        mask = xp.random.rand(*m.shape) >= p
        return m * mask
