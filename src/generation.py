from typing import Literal
import numpy as np
from numpy import typing as npt

from .utils import (
    normalize_vector,
    split_points,
    vector_to_coordinate,
)


# ---------- types ----------

# coordinate = npt.NDArray[np.float32, ...]
# datapoint = npt.NDArray[coordinate, np.float32]
# vector = npt.NDArray[npt.NDArray[np.float32, ...], np.float32]
# edge = npt.NDArray[np.int32, np.int32]
# face = npt.NDArray[np.int32, ...]


def sample_at(
    points: npt.NDArray[np.object_],
    query_position: np.ndarray,
    radius: float,
    mode: Literal["coordinate", "vector"],
    origin_point: np.ndarray | None = None,
) -> float:
    """Return value of nearest datapoint within radius.

    Args:
        points (npt.NDArray[np.object_]): Array of datapoints [[coord], value].
        query_position (np.ndarray): Query position. Interpreted as a coordinate
            if mode is "coordinate" or a [direction, magnitude] vector if mode
            is "vector".
        radius (float): Maximum search radius.
        mode (Literal["coordinate", "vector"]): How to interpret query_position.
        origin_point (np.ndarray | None): Origin used when interpreting
            query_position as a vector. Defaults to zeros.

    Returns:
        float: Value of nearest datapoint within radius, or NaN if none.
    """
    coordinates, values = split_points(points)

    if mode == "vector":
        origin_array = (
            np.zeros_like(query_position, dtype=np.float32)
            if origin_point is None
            else np.asarray(origin_point, dtype=np.float32)
        )
        query_position = vector_to_coordinate(query_position, origin_point=origin_array)

    query_position = np.asarray(query_position, dtype=float)
    distances = np.linalg.norm(coordinates - query_position, axis=1)

    nearest_index = int(np.argmin(distances))
    if distances[nearest_index] > radius:
        return float("nan")

    return float(values[nearest_index])


def march_ray_3d(
    points: npt.NDArray[np.object_],
    ray_vector: npt.NDArray[np.object_],
    radius: float,
    step_size: float,
    max_distance: float,
    default_value: float = np.nan,
    origin_point: np.ndarray | None = None,
) -> float:
    """March a ray through space and return the value of the first datapoint hit.

    Args:
        points (npt.NDArray[np.object_]): Array of datapoints [[coord], value].
        ray_vector (npt.NDArray[np.object_]): Ray as [direction, magnitude].
        radius (float): Hit radius around datapoints.
        step_size (float): Distance between consecutive march steps.
        max_distance (float): Maximum march distance along the ray.
        default_value (float, optional): Value returned if no hit is found.
            Defaults to NaN.
        origin_point (np.ndarray | None, optional): Ray origin. Defaults to [0,0,0].

    Returns:
        float: Value of the first datapoint hit, or default_value if no hit occurs.
    """
    direction, _ = normalize_vector(ray_vector)
    direction = np.asarray(direction, dtype=float)
    origin = (
        np.zeros_like(direction)
        if origin_point is None
        else np.asarray(origin_point, dtype=float)
    )

    step_positions = np.arange(0.0, max_distance + 1e-12, step_size, dtype=float)
    for t in step_positions:
        pos = origin + t * direction
        val = sample_at(points, pos, radius=radius, mode="coordinate")
        if np.isfinite(val):
            return val

    return float(default_value)


def apply_dataseries_to_polygon(
    dataseries: npt.NDArray[np.object_],
    polygon: tuple[np.ndarray, np.ndarray, np.ndarray],
    ray_radius: float = 0.5,
    ray_step: float = 0.25,
    default: float = 0.0,
    origin: tuple[float, float, float] | None = None,
    max_march: float | None = None,
) -> tuple[npt.NDArray[np.object_], np.ndarray, np.ndarray]:
    """Assign a value to each polygon vertex via ray marching over a data series.

    Each input vertex is a vector [direction, magnitude]. For each vertex, a ray
    starts at `origin` and marches along its direction. The first
    datapoint within `ray_radius` is sampled and its value is assigned; otherwise
    `default` is used. The output vertex rows are [direction, magnitude, value].

    Args:
        dataseries (npt.NDArray[np.object_]): Data points packed as [[coordinate], value].
        polygon (tuple[np.ndarray, np.ndarray, np.ndarray]): (vectors, edges, faces),
            where vectors are [[direction], magnitude].
        ray_radius (float): Hit radius around data points.
        ray_step (float): March step length.
        max_march (float | None): Max march distance. If None, computed as
            max(||coordinate||) + ray_step from `data_points`.
        default (float): Value used when no hit occurs.
        origin (tuple[float, float, float] | None): Ray origin coordinate.
    Returns:
        tuple[npt.NDArray[np.object_], np.ndarray, np.ndarray]:
            vectors_out, edges_out, faces_out with vectors_out rows
            [direction, magnitude, value].
    """
    vertex_vectors, edge_indices, face_indices = polygon
    vertex_vectors = np.stack(vertex_vectors)  # (V, 2) object array

    if max_march is None:
        coordinates_dp, _ = split_points(dataseries)
        max_march = float(np.max(np.linalg.norm(coordinates_dp, axis=1) + ray_step))

    vectors_out = np.apply_along_axis(
        lambda vertex_vector: np.array(
            [
                vertex_vector[0],
                vertex_vector[1],
                march_ray_3d(
                    dataseries,
                    vertex_vector,
                    radius=ray_radius,
                    step_size=ray_step,
                    max_distance=max_march,
                    default_value=default,
                    origin_point=origin,
                ),
            ],
            dtype=np.object_,
        ),
        axis=1,
        arr=vertex_vectors,
    )

    return vectors_out, edge_indices, face_indices
