# cython: boundscheck=False, wraparound=False, nonecheck=False

import numpy as np
cimport numpy as np
from cython cimport Py_ssize_t


cdef long long _signed_area(long long[:] a, long long[:] b, long long[:] c):
  """Computes (twice) the signed area of the triangle formed by three points."""
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _signed_area_py(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> int:
  """Numpy version of _signed_area for testing only."""
  return _signed_area(a, b, c)


def get_convex_hull(points_np: np.ndarray) -> np.ndarray:
  """Returns the extremal points of the convex hull of a set of 2D points.

  Uses the simple linear-time algorithm of Graham & Yao (1983).

  Assumes at least two points and that the points are sorted by x-coordinate.

  Args:
    points_np: A numpy array of shape (N, 2) with N >= 2 and dtype np.int64
      containing the points to compute the convex hull of.

  Returns:
    A numpy array of shape (M, 2) with 2 <= M <= N, containing the subset of
    points_np that form the extrema of the convex hull.
  """
  cdef:
    Py_ssize_t i, n
    long long[:, :] points
    long long[:, :] hull

  if points_np.dtype != np.int64:
    raise ValueError('Expected points_np to be of type np.int64.')

  # Check ndim first, or the reference to points_np.shape will crash.
  if points_np.ndim != 2:
    raise ValueError(
      f'Expected 2D array for points_np, got array of dim {points_np.ndim}.'
    )

  if points_np.shape[0] < 2 or points_np.shape[1] != 2:
    raise ValueError(
      'Expected 2D array for points_np, got array of shape '
      f'({points_np.shape[0]}, {points_np.shape[1]}).'
    )

  if not np.all(points_np[:-1, 0] <= points_np[1:, 0]):
    raise ValueError('Expected points_np to be sorted by x-coordinate.')

  points = points_np
  # Create a buffer to hold the extremal points, with maximum size N.
  hull_np = np.empty_like(points_np)
  hull = hull_np

  hull[:2] = points[:2]
  n = 2  # Number of extremal points so far.
  for i in range(2, points.shape[0]):
    while n > 1 and _signed_area(hull[n - 2], hull[n - 1], points[i]) >= 0:
      n -= 1
    hull[n] = points[i]
    n += 1

  return hull_np[:n]
