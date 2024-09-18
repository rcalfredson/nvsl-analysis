#cython: language_level=3
#distutils: language = c++
#distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
#cython: profile=True
#cython: linetrace=True
#cython: binding=True

cimport cython
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libc.math cimport atan2, fabs, fmod, nan, sqrt

cdef double PI = 3.141592653589793

cdef in_range(np.ndarray[long, ndim=1] a, double r1, double r2):
    # Filters an array, returning elements within the specified range [r1, r2).
    #
    # Parameters:
    # - a : np.ndarray[long, ndim=1]
    #     A 1D NumPy array of long integers to be filtered.
    # - r1 : double
    #     The lower bound of the range (inclusive).
    # - r2 : double
    #     The upper bound of the range (exclusive).
    #
    # Returns:
    # - np.ndarray[long, ndim=1]
    #     A 1D NumPy array containing elements from `a` that lie within the specified range.
    cdef int n = a.shape[0]
    cdef int idx_count = 0
    cdef np.ndarray[long, ndim=1] result = np.empty(n, dtype=long)
    
    for i in range(n):
        if r1 <= a[i] and a[i] < r2:
            result[idx_count] = a[i]
            idx_count += 1

    return result[:idx_count]

cdef vector[double] ndarray_long_to_vector(np.ndarray[np.longlong_t] arr):
    # Converts a NumPy array of long long integers into a C++ vector of doubles.
    #
    # Parameters:
    # - arr : np.ndarray[np.longlong_t]
    #     A 1D NumPy array of long long integers to be converted.
    #
    # Returns:
    # - vector[double]
    #     A C++ vector containing the double precision floating-point representation of the
    #     elements in `arr`.
    cdef int n = arr.shape[0]
    cdef vector[double] vec = vector[double]()
    for i in range(n):
        vec.push_back(arr[i])
    return vec

cdef vector[double] ndarray_float_to_vector(np.ndarray[double] arr):
    # Converts a NumPy array of doubles into a C++ vector of doubles.
    #
    # Parameters:
    # - arr : np.ndarray[double]
    #     A 1D NumPy array of double precision floating-point numbers to be converted.
    #
    # Returns:
    # - vector[double]
    #     A C++ vector containing the elements in `arr`.
    cdef int n = arr.shape[0]
    cdef vector[double] vec = vector[double]()
    for i in range(n):
        vec.push_back(arr[i])
    return vec

cdef double compute_distance_or_nan(double x1, double y1, double x2, double y2):
    # Computes the Euclidean distance between two points (x1, y1) and (x2, y2), returning NaN if
    # the points are identical.
    #
    # Parameters:
    # - x1, y1 : double
    #     The coordinates of the first point.
    # - x2, y2 : double
    #     The coordinates of the second point.
    #
    # Returns:
    # - double
    #     The Euclidean distance between the two points or NaN if the points are identical.
    cdef double result = euclidean_norm(x1, y1, x2, y2)
    if result == 0:
        return nan("")
    else:
        return result

@cython.boundscheck(False)  # Deactivate bounds checking for optimized C speed
@cython.wraparound(False)   # Deactivate negative index wrapping for optimized C speed
cdef double euclidean_norm(double x1, double y1, double x2, double y2):
    # Computes the Euclidean norm (distance) between two points in 2D space.
    #
    # Parameters:
    # - x1, y1 : double
    #     The coordinates of the first point.
    # - x2, y2 : double
    #     The coordinates of the second point.
    #
    # Returns:
    # - double
    #     The Euclidean distance between the two points.
    cdef double dx = x1 - x2
    cdef double dy = y1 - y2
    return (dx*dx + dy*dy)**0.5

cdef inline int sign(double x):
    # Determines the sign of a number.
    #
    # Parameters:
    # - x : double
    #     The number whose sign is to be determined.
    #
    # Returns:
    # - int
    #     Returns 1 if `x` is positive, -1 if `x` is negative, and 0 if `x` is zero.
    return (x > 0) - (x < 0)

cdef inline double _angleDiff_inline(double theta1, double theta2, bint absVal, bint useRadians):
    """
    Computes the minimal difference between two angles, accounting for circular continuity.
    """
    cdef double delta = normalize_angle(theta1 - theta2, useRadians)
    
    if absVal:
        return fabs(delta)
    else:
        return delta

cdef double angleDiff(double theta1, double theta2, bint absVal, bint useRadians):
    """
    Computes the minimal difference between two angles, optionally returning the absolute value.
    This version is non-inlined for export and general use across modules.
    """
    return _angleDiff_inline(theta1, theta2, absVal, useRadians)

cdef inline double normalize_angle(double angle, bint useRadians):
    """
    Normalizes an angle to the range [-PI, PI) if in radians or [-180, 180) if in degrees.
    """
    if useRadians:
        angle = fmod(angle + PI, 2 * PI)  # First wrap to the range [0, 2PI)
        if angle < 0:
            angle += 2 * PI  # Shift negatives into the positive range
        return angle - PI  # Now adjust to [-PI, PI)
    else:
        angle = fmod(angle + 180, 360)  # First wrap to the range [0, 360)
        if angle < 0:
            angle += 360  # Shift negatives into the positive range
        return angle - 180  # Now adjust to [-180, 180)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef resolveAngles180To360(double[:] theta, double[:] x, double[:] y, int start_index, double min_jump):
    # Resolves angles from a 180-degree system to a 360-degree system based on motion direction
    # and specified constraints.
    #
    # This function is designed to correct angular data that may flip between 180 and 360
    # degrees due to tracking ambiguities, using the motion direction and a minimum jump
    # distance to decide when to flip angles.
    #
    # Parameters:
    # - theta : memoryview of double
    #     A 1D array of angles in degrees that may require adjustment from 180 to 360 degrees.
    # - x, y : memoryview of double
    #     1D arrays representing the x and y coordinates of points corresponding to the angles
    #     in `theta`.
    # - start_index : int
    #     The starting index from which to begin analyzing and adjusting angles.
    # - min_jump : double
    #     The minimum distance between consecutive points that triggers consideration of an
    #     angle flip.
    #
    # Returns:
    # - np.ndarray
    #     An array with the adjusted angles in degrees, resolved into a 360-degree system where
    #     necessary.
    #
    # Note:
    # This function uses a motion direction-based heuristic to adjust angles, which may not be
    # perfect in all scenarios.
    cdef double max_velocity_angle_weight = 0.25
    cdef double velocity_angle_weight = 0.05
    cdef double[:] theta_360 = theta.copy()
    cdef int N = len(theta)
    cdef unsigned char[:, :] state_prev = np.zeros((N - 1, 2), dtype=np.uint8)
    cdef double[2] tmp_cost = [0, 0]
    cdef double[2] cost_prev_new = [0, 0]
    cdef double[2] cost_prev = [0, 0]
    cdef double rad_to_deg_fctr = 180 / PI
    cdef int tloc, s_curr, s_prev
    cdef double x_curr, y_curr, x_prev, y_prev, v_x, v_y, d_center, velocity_angle, w, w_compl
    cdef double theta_curr, theta_prev, cost_curr
    cdef int[2] angle_flip_range = [0, 1]

    for tloc in range(1, N):
        if tloc < start_index + 1:
            continue
        x_curr = x[tloc]
        y_curr = y[tloc]
        x_prev = x[tloc - 1]
        y_prev = y[tloc - 1]
        v_x = x_curr - x_prev
        v_y = y_prev - y_curr
        d_center = sqrt(v_x*v_x + v_y*v_y)
        velocity_angle = 90 - rad_to_deg_fctr * atan2(v_y, v_x)
        if d_center >= min_jump:
            w = 0
        else:
            angle_wt = velocity_angle_weight * d_center
            if angle_wt < max_velocity_angle_weight:
                w = angle_wt
            else:
                w = max_velocity_angle_weight
        w_compl = 1 - w
        for s_curr in angle_flip_range:
            theta_curr = theta_360[tloc] + s_curr * 180
            for s_prev in angle_flip_range:
                theta_prev = theta_360[tloc - 1] + s_prev * 180
                cost_curr = w_compl * _angleDiff_inline(theta_prev, theta_curr, absVal=True, useRadians=False) \
                           + w * _angleDiff_inline(theta_curr, velocity_angle, absVal=True, useRadians=False)
                tmp_cost[s_prev] = cost_prev[s_prev] + cost_curr
            if tmp_cost[0] < tmp_cost[1]:
                s_prev = 0
            else:
                s_prev = 1
            state_prev[tloc - 1, s_curr] = s_prev
            cost_prev_new[s_curr] = tmp_cost[s_prev]
        cost_prev[0] = cost_prev_new[0]
        cost_prev[1] = cost_prev_new[1]
    if cost_prev[0] < cost_prev[1]:
        s_curr = 0
    else:
        s_curr = 1
    if s_curr == 1:
        theta_360[N - 1] += 180
        theta_360[tloc] = normalize_angle(theta_360[tloc], useRadians=False)

    for tloc in range(N - 2, -1, -1):
        s_curr = state_prev[tloc, s_curr]
        if s_curr == 1:
            theta_360[tloc] += 180
            theta_360[tloc] = normalize_angle(theta_360[tloc], useRadians=False)

    return np.asarray(theta_360)
