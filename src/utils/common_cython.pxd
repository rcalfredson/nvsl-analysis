cimport numpy as np
from libcpp.vector cimport vector

cdef double compute_distance_or_nan(double x1, double y1, double x2, double y2)

cdef double euclidean_norm(double x1, double y1, double x2, double y2)

cdef in_range(np.ndarray[np.int64_t, ndim=1] a, double r1, double r2)

cdef vector[double] ndarray_long_to_vector(np.ndarray[np.longlong_t] arr)

cdef vector[double] ndarray_float_to_vector(np.ndarray[double] arr)

cdef int sign(double x)

cdef double angleDiff(double theta1, double theta2, bint absVal, bint useRadians)

cdef double PI
