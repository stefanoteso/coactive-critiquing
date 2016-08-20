"""
When d > n, i.e. the dimension of the examples is greater than the number of 
examples, there are two possibilities:
 - The rank of the matrix X (n x d) is n. In this case I am pretty sure that 
   all the n points are vertices, but I wasn't able to prove it in short time.
   I will add a check for the convex hull vertices also when the rank is n 
   (which may be redunant if my conjecture is right).
 - The rank of X is k < n, thus k is the number of dimensions in which the 
   points form a proper convex hull  (I'm not able to prove this either but
   it seems to work empirically).
This 
"""

import numpy as np
from numpy.linalg import matrix_rank, norm
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull


def euclidean(a, b):
    return norm(a - b)


def convex_hull_vertices(x):
    """Computes the set of vertices of the convex hull of the input points.
    
    Parameters
    ----------
    x : numpy.ndarray
        A dataset of d-dimensional points (shape = (n, d)).
    """
    n, d = x.shape
    if n > d:
        ch = ConvexHull(xs)
    else:
        r = matrix_rank(x)
        rd = min([n - 1, r])
        pca = PCA(n_components=rd).fit(x)
        proj_x = pca.transform(x)
        ch = ConvexHull(proj_x)
    return [x[i] for i in ch.vertices]


def convex_hull_distance(x, p):
    """Computes the euclidean distance of the point p from the convex hull of 
    the points in x.
    
    The distance is computed as the minimum distance from the vertices of the 
    convex hull.
    
    Parameters
    ----------
    x : numpy.ndarray
        A dataset of d-dimensional points (shape = (n, d)).
    p : numpy.ndarray
        A point in R^d.
    """
    vertices = convex_hull_vertices(x)
    return min([euclidean(vertex, p)] for vertex in vertices)
    

