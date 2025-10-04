from typing import Callable
import numpy as np
import numpy.typing as npt

from .spheroid import Spheroid
from .utils import vector_to_coordinate


def scale_map_to_resolution(
    vertices_2d: np.ndarray,
    bounding_box: tuple[float, float],
    resolution: tuple[int, int],
    padding: float = 0.5,
) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Scale and translate the 2D layout so that it fits within the UV resolution,
    with at least `padding` units from the border.

    The layout is uniformly scaled (same factor in u and v) to avoid distortion,
    then translated so that all coordinates lie ≥ padding from the edges.

    Args:
        vertices_2d (np.ndarray): Array shape (V, 2) of layout coordinates.
        bounding_box (tuple[float, float]): (layout_height, layout_width) span.
        resolution (tuple[int, int]): (res_width, res_height) target UV size.
        padding (float): Absolute margin in UV units/pixels to leave on all sides.

    Returns:
        tuple[np.ndarray, tuple[float, float]]:
            - Scaled and translated vertices array shape (V, 2).
            - The bounding box (layout_height, layout_width) after scaling.
    """
    layout_height, layout_width = bounding_box
    res_width, res_height = resolution

    # effective available space after reserving padding margins on all sides
    avail_width = res_width - 2.0 * padding
    avail_height = res_height - 2.0 * padding
    if avail_width <= 0 or avail_height <= 0:
        raise ValueError(f"Padding {padding} is too large for resolution {resolution}")

    if layout_width <= 0 or layout_height <= 0:
        raise ValueError("Bounding box dimensions must be positive for scaling")

    # compute uniform scale factor
    scale_u = avail_width / layout_width
    scale_v = avail_height / layout_height
    scale = float(min(scale_u, scale_v))

    # scale the layout
    uv_scaled = vertices_2d * scale

    # compute the translation so that minimal coordinates land at padding
    min_uv = np.nanmin(uv_scaled, axis=0)
    translate_u = padding - min_uv[0]
    translate_v = padding - min_uv[1]

    uv_translated = uv_scaled + np.array([translate_u, translate_v], dtype=float)
    return uv_translated, (layout_height * scale, layout_width * scale)


def point_in_polygon(point: tuple[float, float], polygon: np.ndarray) -> bool:
    """
    Determine whether a 2D point lies inside (or on the boundary of) a simple polygon,
    using the even-odd (ray crossing) rule.

    The algorithm casts a horizontal ray to the right from the point, and counts how many
    times that ray intersects polygon edges. If the count is odd, the point is inside;
    even → outside.
    Points exactly on an edge or vertex are considered inside (you may adjust this rule).

    Args:
        point: Tuple (u, v) — the 2D point to test.
        polygon: NumPy array of shape (N, 2) giving the vertices of the polygon in order.

    Returns:
        bool: True if the point is inside or on the polygon boundary, False otherwise.
    """
    u, v = point
    num_vertices = polygon.shape[0]
    inside = False

    # Iterate over edges (i → j)
    j = num_vertices - 1
    for i in range(num_vertices):
        ui, vi = polygon[i]
        uj, vj = polygon[j]
        # Check if the horizontal ray at v crosses edge between vertices i,j
        intersects = (vi > v) != (
            vj > v
        ) and (  # edge straddles the horizontal line at v
            u < (uj - ui) * (v - vi) / (vj - vi + 1e-16) + ui
        )
        if intersects:
            inside = not inside
        j = i

    return inside


def inplane_basis(vertices_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centroid = vertices_3d.mean(axis=0)
    centered_vertices = vertices_3d - centroid
    _, _, right_singular_vectors = np.linalg.svd(centered_vertices, full_matrices=False)
    basis_x = right_singular_vectors[0]
    basis_y = right_singular_vectors[1]
    return basis_x, basis_y, centroid


def project_face_to_2d(vertices_3d: np.ndarray) -> np.ndarray:
    basis_x, basis_y, centroid = inplane_basis(vertices_3d)
    normalized_vertices = vertices_3d - centroid
    return np.column_stack(
        (normalized_vertices @ basis_x, normalized_vertices @ basis_y)
    )


def normalize_face_radius(vertices_2d: np.ndarray, target_radius: float) -> np.ndarray:
    centroid = vertices_2d.mean(axis=0)
    centered_vertices = vertices_2d - centroid

    radii = np.linalg.norm(centered_vertices, axis=1)
    r = float(radii.max())
    if r == 0.0:
        return centered_vertices
    return centered_vertices * (target_radius / r)


def rotate_canonical(vertex: np.ndarray) -> np.ndarray:
    target = 0.0 if vertex.shape[0] == 6 else np.pi / 2.0
    arc = np.arctan2(vertex[0, 1], vertex[0, 0])
    delta = target - arc
    cos_delta, sin_delta = np.cos(delta), np.sin(delta)
    rotation = np.array([[cos_delta, -sin_delta], [sin_delta, cos_delta]])
    return vertex @ rotation.T


def translate_to_packed_position(rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    radius = 1.0
    x_in = np.asarray(rows, dtype=float)
    y_in = np.asarray(columns, dtype=float)
    x_out = 1.5 * radius * x_in
    y_out = np.sqrt(3.0) * radius * (y_in - np.floor(x_in / 2.0) + (radius / 2) * x_in)
    return np.column_stack((x_out, y_out))


def optimize_grid_dimensions(total_cells: int) -> tuple[np.ndarray, np.ndarray]:
    side_length = int(np.ceil(np.sqrt(total_cells)))
    while side_length**2 < total_cells:
        side_length += 1
    rows = np.arange(total_cells) % side_length
    cols = np.arange(total_cells) // side_length
    return rows, cols


def map_face_to_2d(
    face_3d_index: int,
    face_3d: np.ndarray,
    vertices_3d: np.ndarray,
    faces_2d: list[np.ndarray],
    vertices_2d: np.ndarray,
    vertex_map: dict[int, int],
    target_coordinates: np.ndarray,
    padding: float = 0.1,
) -> np.ndarray:
    face_3d_vertex_indices = np.asarray(face_3d, dtype=int)
    face_2d_vertices = normalize_face_radius(
        project_face_to_2d(vertices_3d[face_3d_vertex_indices]),
        target_radius=1.0 - padding,
    )

    vertices_2d = np.vstack((vertices_2d, face_2d_vertices))
    face_2d_index_range: tuple[int, int] = (
        len(vertices_2d) - len(face_3d_vertex_indices),
        len(vertices_2d),
    )
    faces_2d.append(np.arange(*face_2d_index_range))

    for vertex_2d_index, vertex_3d_index in zip(
        range(*face_2d_index_range), face_3d_vertex_indices
    ):
        vertex_map[int(vertex_2d_index)] = int(vertex_3d_index)
    vertices_2d[face_2d_index_range[0] : face_2d_index_range[1]] = rotate_canonical(
        vertices_2d[face_2d_index_range[0] : face_2d_index_range[1]],
    )
    vertices_2d[face_2d_index_range[0] : face_2d_index_range[1]] += target_coordinates[
        face_3d_index
    ]
    return vertices_2d


def unwrap_spheroid(
    spheroid: Spheroid, uv_padding: float = 0.1
) -> tuple[np.ndarray, np.ndarray, dict[int, int], tuple[int, int]]:
    faces_3d = spheroid.faces
    face_count = len(faces_3d)
    vertices_3d: npt.NDArray[np.object_] = np.array(
        [vector_to_coordinate(vector) for vector in spheroid.vectors]
    )

    rows, columns = optimize_grid_dimensions(face_count)
    target_coordinates = translate_to_packed_position(rows, columns)

    vertices_2d: npt.NDArray = np.empty((0, 2), dtype=float)
    faces_2d: list[np.ndarray] = []
    vertex_map: dict[int, int] = {}

    for face_3d_index, face_3d in enumerate(faces_3d):
        vertices_2d = map_face_to_2d(
            face_3d_index,
            face_3d,
            vertices_3d,
            faces_2d,
            vertices_2d,
            vertex_map,
            target_coordinates,
            padding=uv_padding,
        )

    min_x = np.min(vertices_2d[:, 0])
    min_y = np.min(vertices_2d[:, 1])
    if min_x < 0 or min_y < 0:
        vertices_2d[:, 0] -= min_x
        vertices_2d[:, 1] -= min_y

    # tuple (height, width)
    bounding_box = (float(np.max(vertices_2d[:, 1])), float(np.max(vertices_2d[:, 0])))

    return np.array(faces_2d, dtype=object), vertices_2d, vertex_map, bounding_box


def scalar_cross_2d(vec_start: np.ndarray, vec_end: np.ndarray) -> float:
    return vec_start[0] * vec_end[1] - vec_start[1] * vec_end[0]


def calculate_mean_value_coordinates(
    point_2d: np.ndarray, polygon_2d: np.ndarray
) -> np.ndarray:
    """
    We calculate the weights of the polygon's vertices in relation to the given 2D point.

    Args:
        point_2d (np.ndarray): The 2D point as (2,) array.
        polygon_2d (np.ndarray): The vertices as (n, 2) array.

    Returns:
        np.ndarray: The weights for each vertex of the polygon.
    """
    tolerance: float = 1e-12
    vertex_count = polygon_2d.shape[0]
    ray_vectors = polygon_2d - point_2d[None, :]  # (n,2)
    ray_lengths = np.linalg.norm(ray_vectors, axis=1)  # (n,)
    nearest_index = int(np.argmin(ray_lengths))
    if ray_lengths[nearest_index] < tolerance:
        weights = np.zeros(vertex_count, dtype=float)
        weights[nearest_index] = 1.0
        return weights

    # on-edge check
    for ray_index, ray in enumerate(ray_vectors):
        ray_start_vector = ray
        ray_end_vector = ray_vectors[(ray_index + 1) % vertex_count]
        if (
            abs(scalar_cross_2d(ray_start_vector, ray_end_vector)) < tolerance
            and np.dot(ray_start_vector, ray_end_vector) <= 0.0
        ):
            interpolation_fraction = ray_lengths[ray_index] / (
                ray_lengths[ray_index] + ray_lengths[(ray_index + 1) % vertex_count]
            )
            weights = np.zeros(vertex_count, dtype=float)
            weights[ray_index] = 1.0 - interpolation_fraction
            weights[(ray_index + 1) % vertex_count] = interpolation_fraction
            return weights

    angles = np.empty(vertex_count, dtype=float)
    for ray_index, ray in enumerate(ray_vectors):
        ray_start_vector = ray
        ray_end_vector = ray_vectors[(ray_index + 1) % vertex_count]
        angles[ray_index] = np.arctan2(
            scalar_cross_2d(ray_start_vector, ray_end_vector),
            np.dot(ray_start_vector, ray_end_vector),
        )

    half_tangents = np.tan(angles / 2.0)
    weights = (np.roll(half_tangents, 1) + half_tangents) / ray_lengths
    summed_weights = float(weights.sum())
    if abs(summed_weights) < tolerance:
        weights[:] = 0.0
        weights[nearest_index] = 1.0
        return weights
    return weights / summed_weights


class Projection:
    """A 2d representation of how a spheroid maps to a UV map."""

    vertices: npt.ArrayLike  # Array of (u,v) coordinates
    faces: npt.ArrayLike  # Array of (n vertex_indices) where n >= 3
    vertex_map: dict[int, int]  # maps vertex index to spheroid vertex index
    bounding_box: tuple[int, int]  # (width, height) in pixels

    def __init__(
        self,
        faces: npt.ArrayLike | None = None,
        vertices: npt.ArrayLike | None = None,
        vertex_map: dict[int, int] | None = None,
        bounding_box: tuple[int, int] | None = None,
    ):
        """Initialize a Projection.

        Args:
            faces (npt.ArrayLike | None): Array of (n vertex_indices) where n >= 3. Defaults to empty array.
            vertices (npt.ArrayLike | None): Array of (u,v) coordinates. Defaults to empty array.
            vertex_map (dict[int, int] | None): Maps vertex index to spheroid vertex index. Defaults to empty dict.
            bounding_box (tuple[int, int] | None): The bounding box of the UV map (width, height). Defaults to None.

        Returns:
            Projection: The resulting Projection object.
        """
        self.faces = faces if faces is not None else np.empty((0,), dtype=object)
        self.vertices = vertices if vertices is not None else np.empty((0, 2))
        self.vertex_map = vertex_map if vertex_map is not None else {}
        self.bounding_box = bounding_box if bounding_box is not None else (1, 1)

    @classmethod
    def from_spheroid(
        cls, spheroid: Spheroid, resolution: tuple[int, int], **configs
    ) -> "Projection":
        """Create a Projection from a Spheroid by projecting its vertices to UV space.

        Args:
            spheroid (Spheroid): The Spheroid to project.
            resolution (tuple[int, int]): The resolution in pixels of the UV map (width, height).

        Returns:
            Projection: The resulting Projection object.
        """
        # unwrap the spheroid, truncated icosahedron of n >= 0 base icosahedron subdivisions
        # Unwraps faces to a non-overlapping non-contiguous 2D layout and maps vertices from (u,v) to (x,y,z) by index
        faces, vertices_2d, vertex_map, bounding_box = unwrap_spheroid(
            spheroid, configs.get("uv_padding", None)
        )
        # scale and rotate the 2d UV coordinates to fit within the given resolution
        vertices, bounding_box = scale_map_to_resolution(
            vertices_2d, bounding_box, resolution
        )

        return cls(
            faces=faces,
            vertices=vertices,
            vertex_map=vertex_map,
            bounding_box=bounding_box,
        )

    def rasterize(
        self, resolution: tuple[int, int]
    ) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        """Rasterize the Projection to pixel coordinates and face indices.

        Args:
            resolution (tuple[int, int]): The resolution in pixels of the UV map (width, height).

        Returns:
            tuple[npt.ArrayLike, npt.ArrayLike]:
                coords: Array of (u,v) pixel coordinates.
                face_indices: Array of face indices corresponding to coords.
        """
        width, height = resolution
        coords = []
        face_indices = []

        for u in range(width):
            for v in range(height):
                # check which face this (u,v) coordinate falls into
                for face_index, face in enumerate(self.faces):
                    polygon = self.vertices[np.array(face)]  # n gon polygon
                    if point_in_polygon((u, v), polygon):
                        coords.append((u, v))
                        face_indices.append(face_index)
                        break
        return np.array(coords, dtype=int), np.array(face_indices, dtype=int)

    def project_to_spheroid(
        self, u: int, v: int, face_index: int, spheroid: Spheroid, **configs
    ) -> tuple[float, float, float]:
        weights_exponent = configs.get("weights_exponent", 1.0)
        scaler = float(configs.get("scaler", 1.0))
        plane_relaxation = float(configs.get("plane_relaxation", 0.0))

        face = self.faces[face_index]
        polygon_2d = self.vertices[np.array(face)]
        pixel_2d = np.array([u, v], dtype=float)

        weights = calculate_mean_value_coordinates(pixel_2d, polygon_2d)

        if weights_exponent != 1.0:
            weights = np.power(np.maximum(weights, 0.0), weights_exponent)
            weights /= weights.sum()

        face_indices = np.array(face, dtype=int)
        vertex_data = [
            spheroid.vectors[self.vertex_map[int(idx)]] for idx in face_indices
        ]
        vertex_directions = np.stack([vd[0] for vd in vertex_data]).astype(float)
        vertex_magnitudes = np.asarray([vd[1] for vd in vertex_data], dtype=float)

        direction_blend = weights @ vertex_directions
        direction_normal = np.linalg.norm(direction_blend)
        if direction_normal > 0.0:
            direction_blend /= direction_normal

        cartesian_blend = (
            (weights * vertex_magnitudes)[:, None] * vertex_directions
        ).sum(axis=0)

        radius_blend = float(np.dot(direction_blend, cartesian_blend))

        if scaler != 1.0:
            point_3d = direction_blend * (scaler * max(radius_blend, 0.0))
        else:
            point_3d = direction_blend * max(radius_blend, 0.0)

        if plane_relaxation != 0.0:
            vertices_3d_for_plane = np.vstack(
                [
                    vector_to_coordinate(spheroid.vectors[self.vertex_map[int(idx)]])
                    for idx in face_indices
                ]
            ).astype(float)
            basis_x, basis_y, centroid = inplane_basis(vertices_3d_for_plane)
            normal_vector = np.cross(basis_x, basis_y)
            normal_magnitude = np.linalg.norm(normal_vector)
            if normal_magnitude > 0.0:
                normal_vector /= normal_magnitude
                offset_vector = point_3d - centroid
                point_on_plane = (
                    point_3d - np.dot(offset_vector, normal_vector) * normal_vector
                )
                point_3d = (
                    1.0 - plane_relaxation
                ) * point_3d + plane_relaxation * point_on_plane

        return float(point_3d[0]), float(point_3d[1]), float(point_3d[2])

    def scale(self, resolution: tuple[float, float]) -> "Projection":
        """Copies the Projection scaled to a new resolution.

        Args:
            resolution (tuple[float, float]): The new resolution (width, height) for the UV map.

        Returns:
            Projection: A new Projection object scaled to the specified resolution.
        """
        vertices, bounding_box = scale_map_to_resolution(
            self.vertices, self.bounding_box, resolution
        )
        return Projection(
            faces=self.faces,
            vertices=vertices,
            vertex_map=self.vertex_map,
            bounding_box=bounding_box,
        )


class UVMap:
    """A UV map representing a texture map for a spheroid."""

    resolution: tuple[int, int]  # (width, height) in pixels
    coords: npt.ArrayLike  # Array of (u,v) coordinates
    face_indices: npt.ArrayLike  # Array of face indices corresponding to coords
    values: npt.ArrayLike  # Array of values corresponding to coords
    projection: Projection  # The Projection used to create this UVMap

    def __init__(
        self,
        resolution: tuple[int, int],
        coords: npt.ArrayLike | None = None,
        face_indices: npt.ArrayLike | None = None,
        values: npt.ArrayLike | None = None,
        projection: Projection | None = None,
    ):
        """Initialize a UVMap.

        Args:
            resolution (tuple[int, int]): The resolution in pixels of the UV map (width, height).
            coords (npt.ArrayLike | None): Array of (u,v) coordinates. Defaults to empty array.
            face_indices (npt.ArrayLike | None): Array of face indices corresponding to coords. Defaults to empty array.
            values (npt.ArrayLike | None): Array of values corresponding to coords. Defaults to empty array.
            projection (Projection | None): The Projection used to create this UVMap. Defaults to None.

        Returns:
            UVMap: The resulting UVMap object.
        """
        self.resolution = resolution
        self.coords = coords if coords is not None else np.empty((0, 2), dtype=int)
        self.face_indices = (
            face_indices if face_indices is not None else np.empty((0,), dtype=int)
        )
        self.values = values if values is not None else np.empty((0,), dtype=float)
        self.projection = projection if projection is not None else Projection()

    @classmethod
    def from_spheroid(
        cls, resolution: tuple[int, int], spheroid: Spheroid, **configs
    ) -> "UVMap":
        """Create a UVMap from a Spheroid by projecting and rasterizing it.

        Args:
            resolution (tuple[int, int]): The resolution in pixels of the UV map (width, height).
            spheroid (Spheroid): The Spheroid to generate the UV map from.
        Returns:
            UVMap: The resulting UVMap object.
        """
        projection = Projection.from_spheroid(spheroid, resolution, **configs)
        # rasterize the spheroid surface to get (u,v) coordinates and face indices
        coords, face_indices = projection.rasterize(resolution)
        return cls(
            resolution, coords=coords, face_indices=face_indices, projection=projection
        )

    def generate_noise(
        self,
        noise_function: Callable[[tuple[int, int, int]], float],
        spheroid: Spheroid,
        **configs,
    ) -> "UVMap":
        """Generate noise values for the UV map using a provided noise function.

        Args:
            noise_function (callable[[tuple[int, int, int]], float]): A wrapped noise function that takes (u,v,face_index) and returns a float value.
            spheroid (Spheroid): The Spheroid to project onto.

        Returns:
            UVMap: The UVMap object with generated noise values.
        """
        noise_values = []
        for (u, v), face_index in zip(self.coords, self.face_indices):
            value = noise_function(
                *self.projection.project_to_spheroid(
                    u, v, face_index, spheroid, **configs
                )
            )
            noise_values.append(value)
        self.values = np.array(noise_values, dtype=float)
        return self

    def scale(self, resolution: tuple[float, float]) -> "UVMap":
        """Copies the UVMap scaled to a new resolution.

        Args:
            resolution (tuple[float, float]): The new resolution (width, height) for the UV map.

        Returns:
            UVMap: A new UVMap object scaled to the specified resolution.
        """
        # Create a new UVMap with the specified resolution
        projection = self.projection.scale(resolution)
        coords, face_indices = projection.rasterize(resolution)
        return UVMap(
            resolution, coords=coords, face_indices=face_indices, projection=projection
        )
