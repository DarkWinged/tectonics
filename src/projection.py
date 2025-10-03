from typing import Callable
import numpy as np
import numpy.typing as npt

from .spheroid import Spheroid
from .utils import compute_barycentric


def scale_map_to_resolution(
    vertices_2d: np.ndarray,
    bounding_box: tuple[float, float],
    resolution: tuple[int, int],
    padding: float = 0.5,
) -> np.ndarray:
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
        np.ndarray: Array (V, 2) of UV coordinates in pixel units, with margin.
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
    return uv_translated


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
        intersects = (
            (vi > v) != (vj > v)  # edge straddles the horizontal line at v
            and (u < (uj - ui) * (v - vi) / (vj - vi + 1e-16) + ui)
        )
        if intersects:
            inside = not inside
        j = i

    return inside


def coords3d_from_vectors(vectors_obj: np.ndarray) -> np.ndarray:
    v = np.stack(vectors_obj)
    dirs = np.stack(v[:, 0]).astype(float)
    mags = np.asarray(v[:, 1], dtype=float)
    return dirs * mags[:, None]


def inplane_basis(points3: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = points3.mean(axis=0)
    p0, p1, p2 = points3[0], points3[1], points3[2]
    ex = p1 - p0
    ex /= np.linalg.norm(ex)
    n = np.cross(p2 - p0, p1 - p0)
    n /= np.linalg.norm(n)
    ey = np.cross(n, ex)
    return ex, ey, c


def face_planar_2d(points3: np.ndarray) -> np.ndarray:
    ex, ey, c = inplane_basis(points3)
    P = points3 - c
    return np.column_stack((P @ ex, P @ ey))


def rotate_canonical(uv: np.ndarray, k: int) -> np.ndarray:
    if k not in (5, 6):
        return uv
    target = np.pi / 6.0 if k == 6 else np.pi / 2.0
    a0 = np.arctan2(uv[0, 1], uv[0, 0])
    d = target - a0
    c, s = np.cos(d), np.sin(d)
    R = np.array([[c, -s], [s, c]])
    return uv @ R.T


def pack_rowcol_to_xy(row: np.ndarray, col: np.ndarray) -> np.ndarray:
    """30° CCW hex packing: ((√3/2)·row, col - floor(row/2) + 0.5·row)."""
    u = np.asarray(row, dtype=float)
    v = np.asarray(col, dtype=float)
    x = (np.sqrt(3.0) / 2.0) * u
    y = v - np.floor(u / 2.0) + 0.5 * u
    return np.column_stack((x, y))


def grid_rows_cols(count: int) -> tuple[np.ndarray, np.ndarray]:
    cols = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / cols))
    r = np.repeat(np.arange(rows), cols)[:count]
    c = np.tile(np.arange(cols), rows)[:count]
    return r, c


def append_face(
    vertices_2d_list: list,
    faces_list: list,
    vertex_map: dict,
    uv_coords: np.ndarray,
    face_indices_3d: np.ndarray,
) -> None:
    start = len(vertices_2d_list)
    vertices_2d_list.extend(uv_coords.tolist())
    new_idx = np.arange(start, start + len(face_indices_3d), dtype=int)
    faces_list.append(new_idx.tolist())
    for ui, vi in zip(new_idx, face_indices_3d):
        vertex_map[int(ui)] = int(vi)


def shift_positive_and_bbox(
    uv_all: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float]]:
    min_xy = uv_all.min(axis=0)
    uv_all = uv_all - min_xy
    max_xy = uv_all.max(axis=0)
    return uv_all, (float(max_xy[1]), float(max_xy[0]))  # (height, width)


def build_faces_object_array(faces_list: list[list[int]]) -> np.ndarray:
    return np.array([np.asarray(f, dtype=int) for f in faces_list], dtype=object)


def unwrap_spheroid(spheroid: Spheroid):
    faces_src = spheroid.faces  # ordered arrays of vertex indexes
    N = len(faces_src)
    coords3 = coords3d_from_vectors(spheroid.vectors)

    r, c = grid_rows_cols(N)
    centers = pack_rowcol_to_xy(r, c)

    vertices_2d_acc: list[tuple[float, float]] = []
    faces_uv_acc: list[list[int]] = []
    vertex_map: dict[int, int] = {}

    for i, face_idx_list in enumerate(faces_src):
        f = np.asarray(face_idx_list, dtype=int)
        uv = face_planar_2d(coords3[f])
        uv = rotate_canonical(uv, k=len(f))
        uv = uv + centers[i][None, :]
        append_face(vertices_2d_acc, faces_uv_acc, vertex_map, uv, f)

    vertices_2d = np.asarray(vertices_2d_acc, dtype=float)
    vertices_2d, bounding_box = shift_positive_and_bbox(vertices_2d)
    faces = build_faces_object_array(faces_uv_acc)
    return faces, vertices_2d, vertex_map, bounding_box


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
        cls, spheroid: Spheroid, resolution: tuple[int, int]
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
        faces, vertices_2d, vertex_map, bounding_box = unwrap_spheroid(spheroid)
        # scale and rotate the 2d UV coordinates to fit within the given resolution
        vertices = scale_map_to_resolution(vertices_2d, bounding_box, resolution)

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
        self, u: int, v: int, face_index: int, spheroid: Spheroid
    ) -> tuple[float, float, float]:
        """Project a 2D pixel position (u,v) from UV space onto the 3D surface of a spheroid.

        The projection is performed by:
        1. Identifying the polygon (face) in UV space that (u,v) belongs to.
        2. Selecting the three closest vertices of that face in UV space.
        3. Mapping those vertices to their corresponding 3D positions on the target spheroid
            via the stored vertex_map.
        4. Interpolating the pixel's 3D position using barycentric coordinates
            with respect to the triangle formed by those three vertices.

        This provides an approximation of the pixel's 3D location on the spheroid surface.
        Works for convex or concave n-gons (non-self-intersecting).

        Args:
            u (int): Pixel coordinate u in UV space.
            v (int): Pixel coordinate v in UV space.
            face_index (int): Index of the face in the projection that (u,v) belongs to.
            spheroid (Spheroid): The target spheroid to project onto. Must share the same
                face/vertex arrangement as the spheroid used to generate this Projection.

        Returns:
            tuple[float, float, float]: The projected 3D coordinate on the spheroid surface.
        """
        face = self.faces[face_index]
        polygon_2d = self.vertices[np.array(face)]  # (n, 2) polygon in UV space

        # distances from pixel to each vertex in 2D
        pixel_2d = np.array([u, v], dtype=float)
        dists = np.linalg.norm(polygon_2d - pixel_2d, axis=1)

        # indices of the three closest vertices within this face
        face_indices = np.argsort(dists)[:3]
        vertex_indices = np.array(face)[face_indices]
        vertices_2d = polygon_2d[face_indices]

        # get corresponding 3D vertices on the target spheroid
        vertices_3d = np.array(
            [
                spheroid.vertices[self.vertex_map[vertex_index]]
                for vertex_index in vertex_indices
            ],
            dtype=float,
        )

        # compute barycentric coordinates of (u,v) wrt the 2D triangle
        bary = compute_barycentric(pixel_2d, vertices_2d)

        # interpolate 3D position using barycentric weights
        point_3d = np.sum(vertices_3d * bary[:, np.newaxis], axis=0)
        return tuple(point_3d)

    def scale(self, resolution: tuple[float, float]) -> "Projection":
        """Copies the Projection scaled to a new resolution.

        Args:
            resolution (tuple[float, float]): The new resolution (width, height) for the UV map.

        Returns:
            Projection: A new Projection object scaled to the specified resolution.
        """
        vertices = scale_map_to_resolution(self.vertices, resolution)
        return Projection(
            vertices=vertices,
            edges=self.edges,
            faces=self.faces,
            vertex_map=self.vertex_map,
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
    def from_spheroid(cls, resolution: tuple[int, int], spheroid: Spheroid) -> "UVMap":
        """Create a UVMap from a Spheroid by projecting and rasterizing it.

        Args:
            resolution (tuple[int, int]): The resolution in pixels of the UV map (width, height).
            spheroid (Spheroid): The Spheroid to generate the UV map from.
        Returns:
            UVMap: The resulting UVMap object.
        """
        projection = Projection.from_spheroid(spheroid, resolution)
        # rasterize the spheroid surface to get (u,v) coordinates and face indices
        coords, face_indices = projection.rasterize(resolution)
        return cls(
            resolution, coords=coords, face_indices=face_indices, projection=projection
        )

    def generate_noise(
        self,
        noise_function: Callable[[tuple[int, int, int]], float],
        spheroid: Spheroid,
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
                self.projection.project_to_spheroid(u, v, face_index, spheroid)
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
