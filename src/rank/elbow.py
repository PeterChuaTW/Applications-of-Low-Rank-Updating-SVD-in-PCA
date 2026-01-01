"""
Elbow-based rank selection methods:
- Kneedle algorithm
- L-method (two-segment regression)
"""
import numpy as np
import warnings


def kneedle_algorithm(singular_values, sensitivity=1.0):
    """
    Detect elbow using Kneedle algorithm (maximum distance method).

    Returns:
        knee_index (1-based), distances, normalized_distances
    """
    singular_values = np.asarray(singular_values)
    n = len(singular_values)

    if n < 3:
        warnings.warn("Need at least 3 singular values")
        return 1, np.zeros(n), np.zeros(n)

    x = np.arange(1, n + 1)
    x_norm = (x - x[0]) / (x[-1] - x[0]) if x[-1] != x[0] else x

    y = singular_values
    y_norm = (y - y[-1]) / (y[0] - y[-1]) if y[0] != y[-1] else y

    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])

    line_vec = p2 - p1
    line_length = np.linalg.norm(line_vec)

    if line_length == 0:
        warnings.warn("Degenerate case: all values equal")
        return 1, np.zeros(n), np.zeros(n)

    line_unit = line_vec / line_length
    distances = np.zeros(n)

    for i in range(n):
        point = np.array([x_norm[i], y_norm[i]])
        point_vec = point - p1
        cross = point_vec[0] * line_unit[1] - point_vec[1] * line_unit[0]
        distances[i] = abs(cross) * line_length

    normalized_distances = distances / distances.max() if distances.max() > 0 else distances

    threshold = sensitivity * normalized_distances.max()
    candidates = np.where(normalized_distances >= threshold)[0]

    knee_index = (candidates[0] + 1) if len(candidates) > 0 else (np.argmax(distances) + 1)

    return knee_index, distances, normalized_distances


def l_method(singular_values):
    """
    L-method: fit two-segment regression to find elbow.

    Returns:
        elbow_index (1-based), rmse_scores
    """
    singular_values = np.asarray(singular_values)
    n = len(singular_values)

    if n < 4:
        warnings.warn("Need at least 4 singular values")
        return 1, np.array([])

    x = np.log(np.arange(1, n + 1))
    y = np.log(singular_values)

    rmse_scores = np.zeros(n - 3)

    for k in range(1, n - 2):
        x1, y1 = x[:k+1], y[:k+1]
        x2, y2 = x[k+1:], y[k+1:]

        if len(x1) >= 2:
            a1, b1 = np.polyfit(x1, y1, 1)
            rmse1 = np.sqrt(np.mean((y1 - (a1 * x1 + b1))**2))
        else:
            rmse1 = 0

        if len(x2) >= 2:
            a2, b2 = np.polyfit(x2, y2, 1)
            rmse2 = np.sqrt(np.mean((y2 - (a2 * x2 + b2))**2))
        else:
            rmse2 = 0

        rmse_scores[k-1] = rmse1 + rmse2

    elbow_index = np.argmin(rmse_scores) + 2
    return elbow_index, rmse_scores
