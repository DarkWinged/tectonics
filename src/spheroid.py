import numpy as np
import numpy.typing as npt


class Spheroid:
    def __init__(self, radius: float = 1.0, subdivisions: int = 2, dual: bool = True):
        """Initialize a spheroid mesh.

        Args:
            radius (float): Radius of the spheroid. Defaults to 1.0.
            subdivisions (int): Number of subdivisions to apply to base icosahedron.
                More subdivisions = more vertices. Defaults to 2.
            dual (bool): Whether to compute the dual mesh (faces→vertices, vertices→faces).
                Defaults to True.
        """
        self.vectors: npt.NDArray[np.object_]
        self.edges: npt.NDArray[np.int32]
        self.faces: npt.NDArray[np.int32]
        self.icosahedron(radius=radius)
        if subdivisions > 0:
            self.subdivide(subdivisions)
        if dual:
            self.dual()

    def icosahedron(
        self,
        radius: float = 1.0,
    ) -> "Spheroid":
        """Generate an icosahedron as vectors, edges, and faces.

        Args:
            radius (float): Magnitude of vectors (distance from origin).
                Defaults to 1.0.

        Returns:
            Spheroid: representing an icosahedron.
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
        self.vectors = vectors
        self.edges = edges
        self.faces = faces
        return self

    def subdivide(
        self,
        subdivisions: int,
    ) -> "Spheroid":
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
            Spheroid: That has been undergone subdivision.

        Notes:
        - Uses a barycentric grid with P = (s+1)(s+2)/2 points per face.
        - New vertex magnitudes are inherited geometrically: magnitude = ||barycentric blend of original vertex coordinates||.
        - Vertices along shared edges are deduplicated globally (within numeric tolerance).
        """
        vectors_in, faces_in = self.vectors, self.faces
        subdivisions += 1
        if subdivisions <= 0:
            return self

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
        self.vectors = vectors_out
        self.edges = edges_out
        self.faces = faces_mapped
        return self

    def dual(self) -> "Spheroid":
        """Compute the dual of a polygonal mesh.

        Args:
            polygon (np.ndarray): [vectors, edges, faces].
                - vectors: object array [[direction (D,), magnitude], ...]
                - edges: (E,2) int32 array
                - faces: array of index arrays (arbitrary polygons)

        Returns:
            Spheroid: representing the dual mesh.
        """
        vectors, faces = self.vectors, self.faces
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

        self.vectors = dual_vectors
        self.edges = dual_edges
        self.faces = dual_faces

        return self
