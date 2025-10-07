import noise
import numpy as np
import numpy.typing as npt

from .utils import pack_points, split_points


class DataSeries:
    """Class for managing a series of data points with coordinates and values.

    Attributes:
        data (npt.NDArray[np.object_]): Array of shape (N, 2), rows [coord, value].
        configs (dict): Configuration parameters.
    """

    def __init__(self, data: npt.NDArray[np.object_], **configs) -> None:
        """Initialize DataSeries with data points and configurations.

        Args:
            data (npt.NDArray[np.object_]): Array of shape (N, 2), rows [coord, value].
            **configs: Arbitrary configuration metadata.
        """
        self.data = data
        self.configs = {k: v for k, v in configs.items() if v is not None}

    @classmethod
    def one_d_rand(cls, point_count: int, seed: int | None = None) -> "DataSeries":
        """Generate random 1D data as [[x], value].
        Args:
            point_count (int): Number of points to generate.
            seed (int | None): Random seed. Defaults to None.
        Returns:
            DataSeries: Instance containing array of shape (N, 2), with rows [[x], value].
        """
        rng = np.random.default_rng(seed)
        coordinates = np.arange(point_count, dtype=float)[:, None]
        values = rng.integers(0, 10, point_count, dtype=np.int32)
        return cls(pack_points(coordinates, values), seed=seed)

    @classmethod
    def two_d_rand(
        cls, grid_height: int, grid_width: int, seed: int | None = None
    ) -> "DataSeries":
        """Generate random 2D data as [[x, y], value].

        Args:
            grid_height (int): Number of coordinate units along the y-axis.
            grid_width (int): Number of coordinate units along the x-axis.
            seed (int | None): Random seed. Defaults to None.

        Returns:
            DataSeries: Instance containing array of shape (N, 2), with rows [[x, y], value].
        """
        rng = np.random.default_rng(seed)
        y_coords, x_coords = np.meshgrid(
            np.arange(grid_height, dtype=float),
            np.arange(grid_width, dtype=float),
            indexing="ij",
        )
        coordinates = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1)
        values = rng.integers(0, 10, coordinates.shape[0], dtype=np.int32)
        return cls(pack_points(coordinates, values), seed=seed)

    @classmethod
    def three_d_rand(
        cls, grid_height: int, grid_width: int, grid_depth: int, seed: int | None = None
    ) -> "DataSeries":
        """Generate random 3D data as [[x, y, z], value].

        Args:
            grid_height (int): Number of coordinate units along the y-axis.
            grid_width (int): Number of coordinate units along the x-axis.
            grid_depth (int): Number of coordinate units along the z-axis.
            seed (int | None): Random seed. Defaults to None.

        Returns:
            DataSeries: Instance containing array of shape (N, 2), with rows [[x, y, z], value].
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
        return cls(pack_points(coordinates, values), seed=seed)

    @classmethod
    def one_d_perlin(
        cls,
        length: int,
        seed: int | None = None,
        resolution: float = 1.0,
        scale: float = 10.0,
        translation: float = 0.0,
        repeat: int | None = None,
    ) -> "DataSeries":
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
            DataSeries: Instance containing array of shape (N, 2), with rows [[x], value].
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

        data = pack_points(coords, values.astype(float))
        return cls(
            data,
            seed=seed,
            resolution=resolution,
            scale=scale,
            translation=translation,
            repeat=repeat,
        )

    @classmethod
    def two_d_perlin(
        cls,
        height: int,
        width: int,
        seed: int | None = None,
        resolution: float = 1.0,
        scale: float = 10.0,
        translation: tuple[float, float] | None = None,
        repeat: tuple[int, int] | None = None,
    ) -> "DataSeries":
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
            DataSeries: Instance containing array of shape (N, 2), with rows [[x, y], value].
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

        data = pack_points(coords, values.astype(float))
        return cls(
            data,
            seed=seed,
            resolution=resolution,
            scale=scale,
            translation=translation,
            repeat=repeat,
        )

    @classmethod
    def three_d_perlin(
        cls,
        height: int,
        width: int,
        depth: int,
        seed: int | None = None,
        resolution: float = 1.0,
        scale: float = 10.0,
        translation: tuple[float, float, float] | None = None,
        repeat: tuple[int, int, int] | None = None,
    ) -> "DataSeries":
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
            DataSeries: Instance containing array of shape (N, 3), with rows [[x, y, z], value].
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

        data = pack_points(coords, values.astype(float))
        return cls(
            data,
            seed=seed,
            resolution=resolution,
            scale=scale,
            translation=translation,
            repeat=repeat,
        )

    def translate_data(self, translation_vector: np.ndarray) -> "DataSeries":
        """Translate a series of datapoints by a vector.
        Args:
            points (npt.NDArray[np.object_]): Input [coord, value].
            translation_vector (np.ndarray): Translation vector [n,...].

        Returns:
            DataSeries: Instance containing translated [coord, value].
        """
        coords, values = split_points(self.data)
        translated_coords = coords + translation_vector
        self.data = pack_points(translated_coords, values)
        return self

    def cull_below_threshold(self, threshold: float) -> "DataSeries":
        """Remove points below a value threshold.

        Args:
            points (npt.NDArray[np.object_]): Input [coord, value].
            threshold (float): Minimum allowed value.

        Returns:
            npt.NDArray[datapoint]: Filtered array.
        """
        points = self.data
        values = np.array(points[:, 1], dtype=float)
        self.data = points[values >= threshold]
        return self

    def cull_within_radius(
        self,
        radius: float,
        center_point: np.ndarray | None = None,
    ) -> "DataSeries":
        """Remove points within a spherical radius of a center.

        Args:
            points (npt.NDArray[np.object_]): Input [coord, value].
            radius (float): Exclusion radius.
            center_point (np.ndarray | None): Center coordinate. Defaults to mean.

        Returns:
            npt.NDArray[datapoint]: Filtered array.
        """
        points = self.data

        coords, _ = split_points(points)

        if center_point is None:
            center_point = coords.mean(axis=0)
        else:
            center_point = np.asarray(center_point, dtype=float)
        # spherical distances
        distances = np.linalg.vector_norm(coords - center_point, axis=1)
        keep_mask = distances > radius
        self.data = points[keep_mask]
        return self
