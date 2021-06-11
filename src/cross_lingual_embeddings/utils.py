import numpy

try:
    import cupy
except ImportError:
    cupy = None


def supports_cupy():
    """Initialize CUDA support

    Returns:
        obj: cupy package

    """
    return cupy is not None


def get_cupy():
    """Initialize CUDA support

    Returns:
        obj: cupy package

    """
    return cupy


def get_array_module(x):
    """Initialize cupy support if available, if not, use numpy

        Returns:
            obj: cupy or numpy package

    """
    if cupy is not None:
        return cupy.get_array_module(x)
    else:
        return numpy


def asnumpy(x):
    """Initialize cupy support if available, if not, use numpy

        Returns:
            obj: cupy or numpy package

    """
    if cupy is not None:
        return cupy.asnumpy(x)
    else:
        return numpy.asarray(x)


def find_nearest_neighbor(src_matrix, trg_matrix, use_batch=False):
    """Find the nearest neighbor, given 2 embedding matrix

    Args:
        src_matrix (array): Source Embedding Matrix
        trg_matrix (array): Target Embedding Matrix
        use_batch (boolean): Using batch, true or false

    Returns:

    """
    xp = get_array_module(src_matrix)
    if use_batch:
        neighbors = big_matrix_multiplication(src_matrix, trg_matrix.transpose(), get_max=True)
    else:
        matrix_mult = xp.dot(src_matrix, trg_matrix.transpose())
        neighbors = xp.argmax(matrix_mult, axis=1).tolist()
    if any(isinstance(i, list) for i in neighbors):
        neighbors = [neighbor[0] for neighbor in neighbors]
    return neighbors


def normalize_matrix(matrix):
    """Normalize Matrix Embedding.

    Args:
        matrix (array): Matrix

    Returns:
        array: Normalized Matrix (axis=1)

    """
    xp = get_array_module(matrix)
    norms = xp.sqrt(xp.sum(xp.square(matrix), axis=1))
    norms[norms == 0] = 1
    norms = norms.reshape(-1, 1)
    matrix /= norms[:]
    return matrix


def check_if_neighbors_match(src_neighbor, trg_neighbor):
    """Check if any source and target neighbors match and return matches

    Args:
        src_neighbor (list): Source Neighbor List
        trg_neighbor (list): Target Neighbor List

    Returns:
        list: Matching of neighbors.

    """
    matching = {}
    for current_neighbor_index in range(len(src_neighbor)):
        # Looking for matches
        if int(trg_neighbor[src_neighbor[current_neighbor_index]]) == current_neighbor_index:
            matching[current_neighbor_index] = src_neighbor[current_neighbor_index]

    return matching


def mean_center(matrix):
    """Center the matrix.

    Args:
        matrix (array): Matrix

    Returns:
        array: Center the matrix.

    """
    xp = get_array_module(matrix)
    avg = xp.mean(matrix, axis=0)
    matrix -= avg
    return matrix


def vecmap_normalize(matrix):
    """Normalize Matrix and center and normalize again.

    Args:
        matrix: Matrix

    Returns:
        array: Normalize + center + normalize Matrix

    """
    matrix = normalize_matrix(matrix)
    matrix = mean_center(matrix)
    matrix = normalize_matrix(matrix)

    return matrix


def topk_mean(m, k, inplace=False):
    """Top-K Mean of samples

    Args:
        m (array): Matrix
        k (int): Take the most k
        inplace: If we need to comvert to numpy/cupy

    Returns:
        int: topk mean

    """
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
    """Dropout elements from matrix.

    Args:
        m (array):  Matrix
        p (int): Probability to randomly drop elements

    Returns:
        array: Matri with dropped elements

    """
    xp = get_array_module(m)
    if p <= 0.0:
        return m
    else:
        mask = xp.random.rand(*m.shape) >= p
        return m * mask


def big_matrix_multiplication(a, b, get_max=False, chunk_size=10000):
    """Do Big matrix multiplication (useful to save memory)

    Args:
        a (array): Left matrix
        b (array): Right matrix
        get_max (boolean): Calculate maximum of each row of resulting matrix
        chunk_size: Chunk Size of matrix multiplication

    Returns:
        array: Matrix Multiplication result

    """
    result = []
    num_iters = a.shape[0] // chunk_size + (0 if a.shape[0] % chunk_size == 0 else 1)
    for i in range(num_iters):
      res_batch = numpy.dot(a[i * chunk_size : (i+1) * chunk_size, :], b)
      if get_max:
          res_batch = numpy.argmax(res_batch, axis=1).tolist()
      result.extend(res_batch)
    return result

