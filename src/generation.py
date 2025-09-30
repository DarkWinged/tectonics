from typing import Literal
import numpy as np
from numpy import typing as npt
import noise

# ---------- types ----------

# coordinate = npt.NDArray[np.float32, ...]
# datapoint = npt.NDArray[coordinate, np.float32]
# vector = npt.NDArray[npt.NDArray[np.float32, ...], np.float32]
# edge = npt.NDArray[np.int32, np.int32]
# face = npt.NDArray[np.int32, ...]

# ---------- helpers ----------


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


def translate_data(
    points: npt.NDArray[np.object_], translation_vector: np.ndarray
) -> npt.NDArray[np.object_]:
    """Translate a series of datapoints by a vector.
    Args:
        points (npt.NDArray[np.object_]): Input [coord, value].
        translation_vector (np.ndarray): Translation vector [n,...].

    Returns:
        npt.NDArray[np.object_]: Translated [coord, value].
    """
    coords, values = split_points(points)
    translated_coords = coords + translation_vector
    return pack_points(translated_coords, values)


# ---------- random generators ----------


def one_d_rand(point_count: int, seed: int | None = None) -> npt.NDArray[np.object_]:
    """Generate random 1D data as [[x], value].
    Args:
        point_count (int): Number of points to generate.
        seed (int | None): Random seed. Defaults to None.
    Returns:
        npt.NDArray[datapoint]: Array of shape (N, 2), with rows [[x], value].
    """
    rng = np.random.default_rng(seed)
    coordinates = np.arange(point_count, dtype=float)[:, None]
    values = rng.integers(0, 10, point_count, dtype=np.int32)
    return pack_points(coordinates, values)


def two_d_rand(
    grid_height: int, grid_width: int, seed: int | None = None
) -> npt.NDArray[np.object_]:
    """Generate random 2D data as [[x, y], value].

    Args:
        grid_height (int): Number of coordinate units along the y-axis.
        grid_width (int): Number of coordinate units along the x-axis.
        seed (int | None): Random seed. Defaults to None.

    Returns:
        npt.NDArray[datapoint]: Array of shape (N, 2), with rows [[x, y], value].
    """
    rng = np.random.default_rng(seed)
    y_coords, x_coords = np.meshgrid(
        np.arange(grid_height, dtype=float),
        np.arange(grid_width, dtype=float),
        indexing="ij",
    )
    coordinates = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1)
    values = rng.integers(0, 10, coordinates.shape[0], dtype=np.int32)
    return pack_points(coordinates, values)


def three_d_rand(
    grid_height: int, grid_width: int, grid_depth: int, seed: int | None = None
) -> npt.NDArray[np.object_]:
    """Generate random 3D data as [[x, y, z], value].

    Args:
        grid_height (int): Number of coordinate units along the y-axis.
        grid_width (int): Number of coordinate units along the x-axis.
        grid_depth (int): Number of coordinate units along the z-axis.
        seed (int | None): Random seed. Defaults to None.

    Returns:
        npt.NDArray[datapoint]: Array of shape (N, 2), with rows [[x, y, z], value].
    """
    z_coords, y_coords, x_coords = np.meshgrid(
        np.arange(grid_depth, dtype=float),
        np.arange(grid_height, dtype=float),
        np.arange(grid_width, dtype=float),
        indexing="ij",
    )
    coordinates = np.stack(
        [x_coords.ravel(), y_coords.ravel(), z_coords.ravel()], axis=1
    )
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 10, coordinates.shape[0], dtype=np.int32)
    return pack_points(coordinates, values)


# ---------- Perlin generators ----------


def one_d_perlin(
    length: int,
    seed: int | None = None,
    resolution: float = 1.0,
    scale: float = 10.0,
    translation: float = 0.0,
    repeat: int | None = None,
) -> npt.NDArray[np.object_]:
    """Generate 1D Perlin noise points.

    Args:
        length (int): Total coordinate length along x.
        seed (int | None): Noise base seed.
        resolution (float): Grid spacing (units per point).
            1.0 → 1-to-1; 0.5 → 2 pts/unit; 2.0 → 1 pt/2 units.
        scale (float): Divisor applied to coords before noise.
        translation (float): Offset applied to x prior to noise.
        repeat (int | None): Period for tiling (repeatx).

    Returns:
        npt.NDArray[datapoint]: [[x], value] rows.
    """
    if resolution <= 0:
        raise ValueError("resolution must be > 0")
    base = 0 if seed is None else seed

    num_points = max(1, int(round(length / resolution)))
    xs = np.linspace(0.0, length, num_points, endpoint=False, dtype=float)[:, None]
    coords = xs + float(translation)
    noise_x = coords[:, 0] / float(scale)

    if repeat is None:
        values = np.vectorize(lambda u: noise.pnoise1(float(u), base=base))(noise_x)
    else:
        values = np.vectorize(
            lambda u: noise.pnoise1(float(u), base=base, repeat=int(repeat))
        )(noise_x)

    return pack_points(coords, values.astype(float))


def two_d_perlin(
    height: int,
    width: int,
    seed: int | None = None,
    resolution: float = 1.0,
    scale: float = 10.0,
    translation: tuple[float, float] | None = None,
    repeat: tuple[int, int] | None = None,
) -> npt.NDArray[np.object_]:
    """Generate 2D Perlin noise points.

    Args:
        height (int): Extent along y.
        width (int): Extent along x.
        seed (int | None): Noise base seed.
        resolution (float): Grid spacing (units per point).
            1.0 → 1-to-1; 0.5 → 2 pts/unit; 2.0 → 1 pt/2 units.
        scale (float): Divisor applied to coords before noise.
        translation (tuple[float, float] | None): (tx, ty) applied to [x, y].
        repeat (tuple[int, int] | None): (repeatx, repeaty).

    Returns:
        npt.NDArray[datapoint]: [[x, y], value] rows.
    """
    if resolution <= 0:
        raise ValueError("resolution must be > 0")
    base = 0 if seed is None else seed
    tx, ty = (0.0, 0.0) if translation is None else translation

    num_x = max(1, int(round(width / resolution)))
    num_y = max(1, int(round(height / resolution)))

    xs = np.linspace(0.0, width, num_x, endpoint=False, dtype=float)
    ys = np.linspace(0.0, height, num_y, endpoint=False, dtype=float)
    xg, yg = np.meshgrid(xs, ys, indexing="xy")

    coords = np.stack([xg.ravel() + tx, yg.ravel() + ty], axis=1)
    nx = coords[:, 0] / float(scale)
    ny = coords[:, 1] / float(scale)

    if repeat is None:
        values = np.vectorize(
            lambda a, b: noise.pnoise2(float(a), float(b), base=base)
        )(nx, ny)
    else:
        rx, ry = int(repeat[0]), int(repeat[1])
        values = np.vectorize(
            lambda a, b: noise.pnoise2(
                float(a), float(b), base=base, repeatx=rx, repeaty=ry
            )
        )(nx, ny)

    return pack_points(coords, values.astype(float))


def three_d_perlin(
    height: int,
    width: int,
    depth: int,
    seed: int | None = None,
    resolution: float = 1.0,
    scale: float = 10.0,
    translation: tuple[float, float, float] | None = None,
    repeat: tuple[int, int, int] | None = None,
) -> npt.NDArray[np.object_]:
    """Generate 3D Perlin noise points.

    Args:
        height (int): Extent along y.
        width (int): Extent along x.
        depth (int): Extent along z.
        seed (int | None): Noise base seed.
        resolution (float): Grid spacing (units per point).
            1.0 → 1-to-1; 0.5 → 2 pts/unit; 2.0 → 1 pt/2 units.
        scale (float): Divisor applied to coords before noise.
        translation (tuple[float, float, float] | None): (tx, ty, tz).
        repeat (tuple[int, int, int] | None): (repeatx, repeaty, repeatz).

    Returns:
        npt.NDArray[datapoint]: [[x, y, z], value] rows.
    """
    if resolution <= 0:
        raise ValueError("resolution must be > 0")
    base = 0 if seed is None else seed
    tx, ty, tz = (0.0, 0.0, 0.0) if translation is None else translation

    num_x = max(1, int(round(width / resolution)))
    num_y = max(1, int(round(height / resolution)))
    num_z = max(1, int(round(depth / resolution)))

    xs = np.linspace(0.0, width, num_x, endpoint=False, dtype=float)
    ys = np.linspace(0.0, height, num_y, endpoint=False, dtype=float)
    zs = np.linspace(0.0, depth, num_z, endpoint=False, dtype=float)
    xg, yg, zg = np.meshgrid(xs, ys, zs, indexing="xy")

    coords = np.stack([xg.ravel() + tx, yg.ravel() + ty, zg.ravel() + tz], axis=1)
    nx = coords[:, 0] / float(scale)
    ny = coords[:, 1] / float(scale)
    nz = coords[:, 2] / float(scale)

    if repeat is None:
        values = np.vectorize(
            lambda a, b, c: noise.pnoise3(float(a), float(b), float(c), base=base)
        )(nx, ny, nz)
    else:
        rx, ry, rz = int(repeat[0]), int(repeat[1]), int(repeat[2])
        values = np.vectorize(
            lambda a, b, c: noise.pnoise3(
                float(a),
                float(b),
                float(c),
                base=base,
                repeatx=rx,
                repeaty=ry,
                repeatz=rz,
            )
        )(nx, ny, nz)

    return pack_points(coords, values.astype(float))


# ---------- filters ----------


def cull_below_threshold(
    points: npt.NDArray[np.object_], threshold: float
) -> npt.NDArray[np.object_]:
    """Remove points below a value threshold.

    Args:
        points (npt.NDArray[np.object_]): Input [coord, value].
        threshold (float): Minimum allowed value.

    Returns:
        npt.NDArray[datapoint]: Filtered array.
    """
    values = np.array(points[:, 1], dtype=float)
    return points[values >= threshold]


def cull_within_radius(
    points: npt.NDArray[np.object_],
    radius: float,
    center_point: np.ndarray | None = None,
) -> npt.NDArray[np.object_]:
    """Remove points within a spherical radius of a center.

    Args:
        points (npt.NDArray[np.object_]): Input [coord, value].
        radius (float): Exclusion radius.
        center_point (np.ndarray | None): Center coordinate. Defaults to mean.

    Returns:
        npt.NDArray[datapoint]: Filtered array.
    """
    coords, _ = split_points(points)

    if center_point is None:
        center_point = coords.mean(axis=0)
    else:
        center_point = np.asarray(center_point, dtype=float)
    # spherical distances
    distances = np.linalg.vector_norm(coords - center_point, axis=1)
    keep_mask = distances > radius
    return points[keep_mask]


# ---------- vector utilities ----------


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


# ---------- sampling and ray marching ----------


def sample_at(
    points: npt.NDArray[np.object_],
    query_position: np.ndarray,
    radius: float,
    mode: Literal["coordinate", "vector"],
    origin_point: np.ndarray | None = None,
) -> npt.NDArray[np.object_] | float:
    """Find nearest datapoint within radius.

    Args:
        points (npt.NDArray[np.object_]): Input [coord, value].
        query_position (np.ndarray): Query position [n,...] or vector.
        radius (float): Search radius.
        mode (Literal["coordinate", "vector"]): Interpretation of query_position.
        origin_point (np.ndarray | None): Reference origin [n,...].

    Returns:
        np.ndarray | float: Nearest coordinate or vector. np.nan if none.
    """
    coordinates, _ = split_points(points)
    if mode == "vector":
        origin_array = (
            np.zeros_like(query_position, dtype=np.float32)
            if origin_point is None
            else np.asarray(origin_point, dtype=np.float32)
        )
        query_position = vector_to_coordinate(query_position, origin_point=origin_array)
    query_position = np.asarray(query_position, dtype=float)
    distances = np.linalg.norm(coordinates - query_position, axis=1)
    masked_distances = np.where(distances <= radius, distances, np.inf)
    nearest_index = int(np.argmin(masked_distances))
    if not np.isfinite(masked_distances[nearest_index]):
        return float("nan")
    nearest_coord = coordinates[nearest_index]
    if mode == "vector":
        origin_array = (
            np.zeros_like(query_position, dtype=np.float32)
            if origin_point is None
            else np.asarray(origin_point, dtype=np.float32)
        )
        return coordinate_to_vector(
            nearest_coord.astype(np.float32), origin_point=origin_array
        )
    return nearest_coord.astype(np.float32)


def march_ray_3d(
    points: npt.NDArray[np.object_],
    ray_vector: npt.NDArray[np.object_],
    radius: float,
    step_size: float,
    max_distance: float,
    default_value: float = np.nan,
    origin_point: np.ndarray | None = None,
) -> float:
    """March a ray and return first value hit.

    Args:
        points: [coord, value].
        ray_vector: [direction, magnitude].
        radius: Hit radius.
        step_size: Step size.
        max_distance: Maximum travel distance.
        default_value: Returned if no hit.
        origin_point: Ray origin. Defaults to [0,0,0] or provided.

    Returns:
        float: Value of first hit, or default_value.
    """
    direction, _ = normalize_vector(ray_vector)
    direction = np.asarray(direction, dtype=float)

    # ray origin
    origin = (
        np.zeros_like(direction)
        if origin_point is None
        else np.asarray(origin_point, dtype=float)
    )

    # step along the ray
    step_positions = np.arange(0.0, max_distance + 1e-12, step_size, dtype=float)
    for t in step_positions:
        pos = origin + t * direction
        nearest = sample_at(points, pos, radius=radius, mode="coordinate")
        if not np.isnan(nearest).any():  # found a point
            coords, values = split_points(points)
            # match index of nearest coordinate
            idx = np.where((coords == nearest).all(axis=1))[0]
            if idx.size > 0:
                return float(values[idx[0]])
    return float(default_value)


def icosahedron(
    radius: float = 1.0,
) -> tuple[npt.NDArray[np.object_], npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """Generate an icosahedron as vectors, edges, and faces.

    Args:
        radius (float): Magnitude of vectors (distance from origin).
            Defaults to 1.0.

    Returns:
        tuple[npt.NDArray[np.object_], npt.NDArray[np.int32], npt.NDArray[np.int32]]:
            - vectors: (12,) object array of [direction, magnitude].
            - edges: (30, 2) int32 array of vertex index pairs.
            - faces: (20, 3) int32 array of vertex index triplets.
    """
    golden_ratio = (1 + np.sqrt(5)) / 2.0

    # 12 unnormalized vertices
    vertices = np.array(
        [
            [-1, golden_ratio, 0],
            [1, golden_ratio, 0],
            [-1, -golden_ratio, 0],
            [1, -golden_ratio, 0],
            [0, -1, golden_ratio],
            [0, 1, golden_ratio],
            [0, -1, -golden_ratio],
            [0, 1, -golden_ratio],
            [golden_ratio, 0, -1],
            [golden_ratio, 0, 1],
            [-golden_ratio, 0, -1],
            [-golden_ratio, 0, 1],
        ],
        dtype=np.float32,
    )

    # Normalize to unit directions
    directions = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

    # Store as [direction, magnitude]
    vectors = np.empty((len(directions),), dtype=np.object_)
    for i, direction in enumerate(directions):
        vectors[i] = np.array(
            [direction.astype(np.float32), float(radius)], dtype=np.object_
        )

    # 20 triangular faces
    faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int32,
    )

    # Unique edges from faces
    edge_pairs = np.vstack(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ]
    )
    edges = np.unique(np.sort(edge_pairs, axis=1), axis=0)

    # Pack result into object array
    return vectors, edges, faces


def subdivide_polygon(
    polygon: npt.NDArray[np.object_],
    subdivisions: int,
) -> tuple[npt.NDArray[np.object_], npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """Subdivide all triangular faces of a polygon in one pass (no loops/recursion).

    Args:
      polygon (npt.NDArray[np.object_]):
        Array-like [[vectors], [edges], [faces]] where:
          - vectors: npt.NDArray[np.object_] of shape (V, 2), rows [[direction (D,)], magnitude]
          - edges: np.ndarray[int] of shape (E, 2)
          - faces: np.ndarray[int] of shape (F, 3) (assumed triangles)
      subdivisions (int):
        Number of uniform barycentric subdivisions per face (s >= 1).

    Returns:
      tuple[npt.NDArray[np.object_], npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        [[vectors], [edges], [faces]] after subdivision.
        - vectors: npt.NDArray[np.object_] of shape (V', 2), rows [[direction (D,)], magnitude]
        - edges: np.ndarray[int] of shape (E', 2)
        - faces: np.ndarray[int] of shape (K', 3)

    Notes:
      - Uses a barycentric grid with P = (s+1)(s+2)/2 points per face.
      - New vertex magnitudes are inherited geometrically: magnitude = ||barycentric blend of original vertex coordinates||.
      - Vertices along shared edges are deduplicated globally (within numeric tolerance).
    """
    vectors_in, edges_in, faces_in = polygon
    subdivisions += 1
    if subdivisions <= 0:
        return vectors_in.copy(), edges_in.copy(), faces_in.copy()

    # Unpack input
    vectors_in: npt.NDArray[np.object_] = vectors_in.copy()
    faces_in: np.ndarray = faces_in.copy()

    vectors_in = np.stack(vectors_in)  # (V, 2)

    directions_in = np.stack(vectors_in[:, 0]).astype(float)  # (V, D)
    magnitudes_in = vectors_in[:, 1].astype(float)  # (V,)
    coords_in = directions_in * magnitudes_in[:, None]  # (V, D)

    # Triangle vertex coordinates per face
    tri_coords = coords_in[faces_in]  # (F, 3, D)
    F, _, D = tri_coords.shape
    s = int(subdivisions)

    # ----- Barycentric grid (template for one face) -----
    I, J = np.meshgrid(np.arange(s + 1), np.arange(s + 1), indexing="ij")
    mask = (I + J) <= s
    ii = I[mask]  # (P,)
    jj = J[mask]  # (P,)
    kk = s - (ii + jj)  # (P,)
    P = ii.size
    bary = (np.stack([ii, jj, kk], axis=1) / float(s)).astype(float)  # (P, 3)

    # Local index grid for template points
    idx_grid = np.full((s + 1, s + 1), -1, dtype=int)
    idx_grid[mask] = np.arange(P)

    # ----- Template faces (triangulation) -----
    I0, J0 = np.meshgrid(np.arange(s), np.arange(s), indexing="ij")
    mask_cell = (I0 + J0) <= (s - 1)  # cells with one lower tri
    mask_up = (I0 + J0) <= (s - 2)  # cells with upper tri

    # lower triangles: (i,j), (i+1,j), (i,j+1)
    a = idx_grid[I0[mask_cell], J0[mask_cell]]
    b = idx_grid[I0[mask_cell] + 1, J0[mask_cell]]
    c = idx_grid[I0[mask_cell], J0[mask_cell] + 1]
    tri_lower = np.stack([a, b, c], axis=1)  # (T1, 3)

    # upper triangles: (i+1,j), (i+1,j+1), (i,j+1)
    b2 = idx_grid[I0[mask_up] + 1, J0[mask_up]]
    d2 = idx_grid[I0[mask_up] + 1, J0[mask_up] + 1]
    c2 = idx_grid[I0[mask_up], J0[mask_up] + 1]
    tri_upper = np.stack([b2, d2, c2], axis=1)  # (T2, 3)

    # ----- Template edges (three directions) -----
    # Along +i: (i,j)->(i+1,j)
    e0a = idx_grid[I0[mask_cell], J0[mask_cell]]
    e1a = idx_grid[I0[mask_cell] + 1, J0[mask_cell]]
    edges_a = np.stack([e0a, e1a], axis=1)

    # Along +j: (i,j)->(i,j+1)
    e0b = idx_grid[I0[mask_cell], J0[mask_cell]]
    e1b = idx_grid[I0[mask_cell], J0[mask_cell] + 1]
    edges_b = np.stack([e0b, e1b], axis=1)

    # Diagonal: (i+1,j)->(i,j+1)
    e0c = idx_grid[I0[mask_up] + 1, J0[mask_up]]
    e1c = idx_grid[I0[mask_up], J0[mask_up] + 1]
    edges_c = np.stack([e0c, e1c], axis=1)

    edges_templ = np.vstack([edges_a, edges_b, edges_c])  # (E_t, 2)

    # ----- New coordinates for all faces (barycentric apply) -----
    # tensordot over the 3 bary weights -> (P, F, D), then transpose to (F, P, D)
    new_coords_fp = np.tensordot(bary, tri_coords, axes=([1], [1])).transpose(
        1, 0, 2
    )  # (F, P, D)

    # Magnitudes inherited geometrically; directions normalized
    # new_mags_fp = np.linalg.norm(new_coords_fp, axis=2)  # (F, P)
    # # Avoid div by zero (degenerate case)
    # safe = np.where(new_mags_fp == 0.0, 1.0, new_mags_fp)
    # new_dirs_fp = new_coords_fp / safe[..., None]  # (F, P, D)

    # ----- Global vertex deduplication -----
    coords_flat = new_coords_fp.reshape(-1, D)  # (F*P, D)
    # Quantize for stable uniqueness
    q = np.round(coords_flat, decimals=12)
    q_view = q.view([("", q.dtype)] * D).reshape(-1)
    _, unique_idx, inverse_idx = np.unique(
        q_view, return_index=True, return_inverse=True
    )

    coords_unique = coords_flat[unique_idx]  # (V', D)
    mags_unique = np.linalg.norm(coords_unique, axis=1)  # (V',)
    dirs_unique = (
        coords_unique / np.where(mags_unique == 0.0, 1.0, mags_unique)[:, None]
    )

    # Pack vectors [[direction], magnitude]
    Vp = coords_unique.shape[0]
    vectors_out = np.empty((Vp, 2), dtype=np.object_)
    # column-wise assignment (object arrays)
    vectors_out[:, 0] = [d for d in dirs_unique]
    vectors_out[:, 1] = mags_unique.astype(float)

    # Map template indices to global unique vertex indices
    map_face = inverse_idx.reshape(F, P)  # (F, P)

    # Faces (lower + upper), replicated over all faces
    t1 = tri_lower.shape[0]
    t2 = tri_upper.shape[0]
    tri_all_local = np.vstack([tri_lower, tri_upper])  # (t1+t2, 3)
    pick = tri_all_local.ravel()  # ((t1+t2)*3,)
    faces_mapped = map_face[:, pick].reshape(F, t1 + t2, 3).reshape(-1, 3)

    # Edges: replicate template and deduplicate globally
    e_pick = edges_templ.ravel()  # (E_t*2,)
    edges_face = map_face[:, e_pick].reshape(F, -1, 2).reshape(-1, 2)  # (F*E_t, 2)
    edges_sorted = np.sort(edges_face, axis=1)
    e_view = edges_sorted.view([("", edges_sorted.dtype)] * 2).reshape(-1)
    edges_out = edges_sorted[
        np.unique(e_view, return_index=True)[1]
    ]  # unique global edges

    # ----- Package output -----
    out = np.empty(3, dtype=np.object_)
    out[0] = vectors_out
    out[1] = edges_out
    out[2] = faces_mapped
    return out


def dual_polygon(polygon: np.ndarray) -> np.ndarray:
    """Compute the dual of a polygonal mesh.

    Args:
        polygon (np.ndarray): [vectors, edges, faces].
            - vectors: object array [[direction (D,), magnitude], ...]
            - edges: (E,2) int32 array
            - faces: array of index arrays (arbitrary polygons)

    Returns:
        np.ndarray: [dual_vectors, dual_edges, dual_faces]
    """
    vectors, _, faces = polygon
    vectors = np.stack(vectors)
    directions = np.stack(vectors[:, 0]).astype(float)
    magnitudes = vectors[:, 1].astype(float)
    coords = directions * magnitudes[:, None]

    # --- dual vertices (face centroids) ---
    face_coords = np.array([coords[f].mean(axis=0) for f in faces])
    dual_mags = np.linalg.norm(face_coords, axis=1)
    safe_mags = np.where(dual_mags == 0, 1.0, dual_mags)
    dual_dirs = face_coords / safe_mags[:, None]

    dual_vectors = np.empty((len(faces), 2), dtype=object)
    dual_vectors[:, 0] = [d.astype(np.float32) for d in dual_dirs]
    dual_vectors[:, 1] = dual_mags.astype(float)

    # --- build vertex→incident faces map ---
    vertex_to_faces = [[] for _ in range(len(coords))]
    for f_idx, f in enumerate(faces):
        for v in f:
            vertex_to_faces[v].append(f_idx)

    # --- also build edge→face map ---
    edge_to_faces = {}
    for f_idx, f in enumerate(faces):
        for i_idx, i in enumerate(f):
            a, b = int(i), int(f[(i_idx + 1) % len(f)])
            key = tuple(sorted((a, b)))
            edge_to_faces.setdefault(key, []).append(f_idx)

    # --- dual faces (per original vertex) ---
    dual_faces = []
    for v_idx, incident_faces in enumerate(vertex_to_faces):
        if len(incident_faces) < 3:
            continue

        # walk around the vertex via edges
        ordered = []
        # pick an arbitrary starting face
        current_face = incident_faces[0]
        used = set()
        while True:
            ordered.append(current_face)
            used.add(current_face)

            # find next face around v_idx sharing an edge
            next_face = None
            f = faces[current_face]
            for i_idx, i in enumerate(f):
                a, b = i, f[(i_idx + 1) % len(f)]
                if v_idx not in (a, b):
                    continue
                key = tuple(sorted((a, b)))
                for nbr in edge_to_faces.get(key, []):
                    if (
                        nbr != current_face
                        and nbr not in used
                        and nbr in incident_faces
                    ):
                        next_face = nbr
                        break
                if next_face is not None:
                    break

            if next_face is None:
                break
            current_face = next_face
            if current_face == ordered[0]:
                break

        if len(ordered) >= 3:
            dual_faces.append(np.array(ordered, dtype=np.int32))

    # --- dual edges ---
    edge_set = set()
    for f in dual_faces:
        for i, _ in enumerate(f):
            a, b = int(f[i]), int(f[(i + 1) % len(f)])
            edge_set.add(tuple(sorted((a, b))))
    dual_edges = np.array(list(edge_set), dtype=np.int32)

    return np.array(
        [dual_vectors, dual_edges, np.array(dual_faces, dtype=object)], dtype=object
    )


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
