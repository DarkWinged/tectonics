import numpy as np
import numpy.typing as npt


def pack_points(coordinates: np.ndarray, values: np.ndarray) -> npt.NDArray[np.object_]:
    """Pack coordinates and values into [coord, value] rows.

    Args:
        coordinates (np.ndarray): Coordinates of shape (N, D).
        values (np.ndarray): Values of shape (N,).

    Returns:
        array[datapoint]: Object array of shape (N, 2), rows [coord, value].
    """
    n = coordinates.shape[0]
    result = np.empty((n, 2), dtype=object)
    result[:, 0] = [c for c in coordinates]  # each row is an ndarray (D,)
    result[:, 1] = values
    return result


def split_points(points: npt.NDArray[np.object_]) -> tuple[np.ndarray, np.ndarray]:
    """Split [coord, value] array into separate arrays.

    Args:
        points (npt.NDArray[np.object_]): Input array of shape (N, 2).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            coordinates: float array of shape (N, D).
            values: float array of shape (N,).
    """
    coords = np.stack(points[:, 0], axis=0).astype(float)  # (N, D)
    values = points[:, 1].astype(float)  # (N,)
    return coords, values


def coordinate_to_vector(
    coordinate: np.ndarray, origin_point: np.ndarray | None = None
) -> npt.NDArray[np.object_]:
    """Convert coordinate to [direction, magnitude].

    Args:
        coordinate (np.ndarray): Coordinate [n,...].
        origin_point (np.ndarray | None): Origin [n,...]. Defaults to zeros.

    Returns:
        npt.NDArray[vector]: [direction, magnitude].
    """
    coordinate = np.asarray(coordinate, dtype=np.float32)
    origin_array = (
        np.zeros_like(coordinate)
        if origin_point is None
        else np.asarray(origin_point, dtype=np.float32)
    )
    difference = coordinate - origin_array
    magnitude = float(np.linalg.norm(difference))
    if magnitude == 0.0:
        raise ValueError("Cannot convert origin coordinate to vector.")
    direction = difference / magnitude
    return np.array([direction, magnitude], dtype=object)


def vector_to_coordinate(
    vector: npt.NDArray[np.object_], origin_point: np.ndarray | None = None
) -> np.ndarray:
    """Convert [direction, magnitude] to coordinate.

    Args:
        vector (npt.NDArray[np.object_]): [direction, magnitude].
        origin_point (np.ndarray | None): Origin [n,...]. Defaults to zeros.

    Returns:
        np.ndarray[coordinate]: Coordinate [n,...].
    """
    direction, magnitude = vector
    direction_array = np.asarray(direction, dtype=np.float32)
    magnitude_value = float(magnitude)
    origin_array = (
        np.zeros_like(direction_array)
        if origin_point is None
        else np.asarray(origin_point, dtype=np.float32)
    )
    return origin_array + direction_array * magnitude_value


def scale_vector(
    vector: npt.NDArray[np.object_], scale_factor: float
) -> npt.NDArray[np.object_]:
    """Scale vector magnitude by factor.

    Args:
        vector (npt.NDArray[np.object_]): [direction, magnitude].
        scale_factor (float): Scale factor.

    Returns:
        npt.NDArray[vector]: Scaled [direction, magnitude].
    """
    direction, magnitude = vector
    return np.array(
        [
            np.asarray(direction, dtype=np.float32),
            float(magnitude) * float(scale_factor),
        ],
        dtype=object,
    )


def normalize_vector(vector: npt.NDArray[np.object_]) -> npt.NDArray[np.object_]:
    """Normalize vector to unit length and magnitude = 1.

    Args:
        vector (npt.NDArray[np.object_]): [direction, magnitude].

    Returns:
        npt.NDArray[vector]: Normalized [direction, 1.0].
    """
    direction, _ = vector
    direction_array = np.asarray(direction, dtype=np.float32)
    norm_value = float(np.linalg.norm(direction_array))
    if norm_value == 0.0:
        raise ValueError("Cannot normalize zero-length vector.")
    return np.array([direction_array / norm_value, 1.0], dtype=object)


def get_vector_magnitude(vector: npt.NDArray[np.object_]) -> float:
    """Get magnitude from vector.

    Args:
        vector (npt.NDArray[np.object_]): [direction, magnitude].

    Returns:
        float: Magnitude.
    """
    return float(vector[1])


def compute_barycentric(P: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    """Compute barycentric weights (a, B, y) of point P w.r.t. 2D triangle (A, B, C).

    Given triangle vertices A, B, C ∈ ℝ² and a query point P ∈ ℝ², compute
    weights (a, B, y) such that:
        P = a·A + B·B + y·C,   with   a + B + y = 1.

    Degenerate triangle fallback: return [1/3, 1/3, 1/3].

    Args:
      P: (2,) array, the query point.
      triangle: (3, 2) array, triangle vertices ordered as [A, B, C].

    Returns:
      (3,) float array [a, B, y].

    References:
      - Scratchapixel: Barycentric coordinates
        https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/barycentric-coordinates.html
      - Wikipedia: Barycentric coordinate system (triangle)
        https://en.wikipedia.org/wiki/Barycentric_coordinate_system
    """
    A, B, C = triangle

    v0 = C - A
    v1 = B - A
    v2 = P - A

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)

    denom = d00 * d11 - d01 * d01
    if np.isclose(denom, 0.0):
        return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)

    B = (d11 * d20 - d01 * d21) / denom
    y = (d00 * d21 - d01 * d20) / denom
    a = 1.0 - B - y

    return np.array([a, B, y], dtype=float)
