# well_contact.pyx
# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
import cython
import math
import numpy as np
cimport numpy as cnp
from libc.math cimport cos, sin, sqrt, fabs

ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.uint8_t BOOL_t

def detect_well_contacts_edge_or_center(
        double[::1] x,
        double[::1] y,
        double[::1] rot_angles_deg,
        double[::1] semimaj_ax,
        double[::1] semimin_ax,
        cnp.ndarray[BOOL_t, ndim=1] lost,
        tuple wells,            # [(cx, cy), ...] from arenaWells
        double well_radius,
        str ref_mode="edge"):
    """
    Returns a bool array `well_contact` (1 = inside/overlap) per frame.
    Frames with NaN x are copied through as NaN.
    """
    cdef Py_ssize_t n = x.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=1] well_contact = np.empty(n, dtype=np.float64)

    cdef Py_ssize_t i, j
    cdef double xe, ye, a, b, theta
    cdef double cx, cy
    cdef cython.bint in_contact, enter_hit, exit_hold
    cdef double r_enter, r_exit, r_enter2, r_exit2

    # initialize hysteresis thresholds
    in_contact = False
    r_enter = well_radius
    r_exit = well_radius + 1.0
    r_enter2 = r_enter * r_enter
    r_exit2 = r_exit * r_exit

    for i in range(n):
        xe = x[i]
        ye = y[i]
        if lost[i] or np.isnan(xe):           # lost frame → propagate NaN
            well_contact[i] = np.nan
            continue
        theta = _deg_to_rad_world(rot_angles_deg[i])
        a = semimaj_ax[i]          # width/2 in ellipse frame
        b = semimin_ax[i]          # height/2 in ellipse frame

        enter_hit = False
        exit_hold = False
        for j in range(len(wells)):
            cx, cy = wells[j]
            if ref_mode == "center":
                # simple point‑in‑circle
                dc2 = (xe-cx)*(xe-cx) + (ye-cy)*(ye-cy)
                if dc2 <= r_enter2:
                    enter_hit = True
                    exit_hold = True
                    break
                elif dc2 <= r_exit2:
                    exit_hold = True
            else:  # "edge"
                # "enter" overlap at r_enter
                if _ellipse_circle_overlap_edge(
                        xe, ye, a, b, theta, cx, cy, r_enter, i):
                    enter_hit = True
                    exit_hold = True
                    break
                # "exit" overlap at r_exit
                elif _ellipse_circle_overlap_edge(
                    xe, ye, a, b, theta, cx, cy, r_exit, i
                ):
                    exit_hold = True
        # hysteresis update
        if in_contact:
            in_contact = exit_hold
        else:
            in_contact = enter_hit
        well_contact[i] = in_contact
    return well_contact

cdef inline double _deg_to_rad_world(double rot_angle_deg):
    """
    Your rot_angles use: 0° = pointing 'north', +deg = CW.
    Convert to standard CCW-from-+x radians.
    """
    return (3.141592653589793 / 180.0) * (rot_angle_deg - 90)

@cython.cfunc
@cython.inline
def _to_ellipse_frame(
        double x, double y,
        double xe, double ye,
        double theta_rad,
        double* xp, double* yp):
    """
    World → ellipse-aligned frame (ellipse center at origin, axes aligned).
    """
    cdef double dx = x - xe
    cdef double dy = y - ye
    cdef double c = cos(theta_rad)
    cdef double s = sin(theta_rad)
    xp[0] =  c * dx + s * dy
    yp[0] = -s * dx + c * dy


cpdef _ellipse_circle_overlap_edge(
        double xe, double ye,   # ellipse centre
        double a,  double b,    # semi‑axes: a = width/2, b = height/2
        double theta,           # radians, CCW from +x
        double cx, double cy, double r,
        int frame,
        double tol = 1e-9):
    """
    Quartic-hybrid exact test (no perimeter sampling).
    Returns 1 if ellipse & circle overlap / touch.
    """
    # ---------- cheap bounding‑circle rejection -----------------------------
    cdef double dx = cx - xe
    cdef double dy = cy - ye
    cdef double max_sep = r + (a if a > b else b)

    if dx*dx + dy*dy > max_sep*max_sep:
        return 0

    # ---------- centre‑containment quick wins -------------------------------
    # Circle‑centre inside ellipse?
    cdef double xp, yp
    _to_ellipse_frame(cx, cy, xe, ye, theta, &xp, &yp)
    eqv = (xp*xp)/(a*a) + (yp*yp)/(b*b)
    if eqv <= 1.0:
        return 1
    # Ellipse centre inside circle?
    if dx*dx + dy*dy <= r*r:
        return 1

    # ---------- quartic intersection test -----------------------------------
    # All variables now in ellipse frame
    cdef double x0 = xp
    cdef double y0 = yp
    cdef double inv_b2 = 1.0 / (b*b)
    cdef double A = a*a
    cdef double R2 = r*r

    # Polynomial coefficients for P(y) = 0 (see derivation in previous answer)
    cdef double C0 = A + x0*x0 + y0*y0 - R2
    cdef double C1 = -2.0 * y0
    cdef double C2 = 1.0 - A*inv_b2
    cdef double V0 = 4.0 * A * x0*x0
    cdef double V2 = -4.0 * A * x0*x0 * inv_b2

    cdef double p4 = C2*C2
    cdef double p3 = 2.0 * C1 * C2
    cdef double p2 = 2.0*C0*C2 + C1*C1 - V2
    cdef double p1 = 2.0 * C0 * C1
    cdef double p0 = C0*C0 - V0

    # Handle degenerate leading coeff
    cdef DTYPE_t[5] coeffs_view = np.asarray([p4, p3, p2, p1, p0], dtype=np.float64)
    cdef int start = 0
    while start < 4 and fabs(coeffs_view[start]) < tol:
        start += 1
    coeffs_all = [p4, p3, p2, p1, p0]
    # drop leading near‑zero ones
    if start >= 5:
        return 0
    coeffs_py = coeffs_all[start:]       # this is a Python list

    for c in coeffs_py:
        if not math.isfinite(c):
            # degenerate geometry or invalid semi-axis → no intersection
            return 0

    # now call numpy.roots on that list
    roots = np.roots(coeffs_py)

    cdef double y, S, x_abs, dx2, dy2
    for val in roots:
        if fabs(val.imag) > 1e-8:
            continue   # discard complex root
        y = val.real
        S = 1.0 - (y*y)*inv_b2
        if S < -tol:
            continue
        if S < 0.0:
            S = 0.0
        x_abs = sqrt(A * S)
        # two symmetric x'
        # (x' - x0)^2 + (y - y0)^2 <= R2 ?
        dx2 = ( x_abs - x0 )
        dy2 =   y - y0
        if dx2*dx2 + dy2*dy2 <= R2 + tol:
            return 1
        dx2 = (-x_abs - x0)
        if dx2*dx2 + dy2*dy2 <= R2 + tol:
            return 1
    return 0
