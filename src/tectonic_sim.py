from dataclasses import dataclass
import numpy as np
from numpy import typing as npt
from queue import PriorityQueue


from src.spheroid import Spheroid


def lonlat_from_cartesian(vertex: npt.ArrayLike) -> np.ndarray:
    xyz = np.asarray(vertex, dtype=float)
    r = np.linalg.norm(xyz)
    longitude = np.arctan2(xyz[1], xyz[0])  # [-π, π]
    latitude = np.arcsin(np.clip(xyz[2] / r, -1.0, 1.0))  # [-π/2, π/2]
    return np.array([longitude, latitude], dtype=float)


def spherical_tangent_basis(point: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    longitude, latitude = lonlat_from_cartesian(point)
    east_unit = np.array([-np.sin(longitude), np.cos(longitude), 0.0], dtype=float)
    north_unit = np.array(
        [
            -np.cos(longitude) * np.sin(latitude),
            -np.sin(longitude) * np.sin(latitude),
            np.cos(latitude),
        ],
        dtype=float,
    )
    return east_unit, north_unit  # unit vectors


def tangent_vector_from_lonlat_rates(
    point: npt.ArrayLike, lonlat_rates: npt.ArrayLike
) -> np.ndarray:
    longitude, latitude = lonlat_from_cartesian(point)
    dlon, dlat = np.asarray(lonlat_rates, dtype=float)
    # ∂r/∂λ = cos(lat) * east_unit,  ∂r/∂φ = north_unit  (unit sphere)
    east_unit, north_unit = spherical_tangent_basis(point)
    return dlon * np.cos(latitude) * east_unit + dlat * north_unit


def parallel_transport_tangent_vector(
    source_point: npt.ArrayLike,
    target_point: npt.ArrayLike,
    vector_at_source: npt.ArrayLike,
) -> np.ndarray:
    a = np.asarray(source_point, dtype=float)
    a /= np.linalg.norm(a)
    b = np.asarray(target_point, dtype=float)
    b /= np.linalg.norm(b)
    axis = np.cross(a, b)
    axis_norm = np.linalg.norm(axis)
    v = np.asarray(vector_at_source, dtype=float)
    if axis_norm < 1e-12:
        return v
    axis /= axis_norm
    angle = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
    c, s = np.cos(angle), np.sin(angle)
    # Rodrigues rotation around the great-circle axis
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1.0 - c)


def geodesic_unit_direction(
    point: npt.ArrayLike, toward_point: npt.ArrayLike
) -> np.ndarray:
    p = np.asarray(point, dtype=float)
    p /= np.linalg.norm(p)
    q = np.asarray(toward_point, dtype=float)
    q /= np.linalg.norm(q)
    tangent = q - np.dot(q, p) * p
    n = np.linalg.norm(tangent)
    if n < 1e-12:
        return np.zeros(3, dtype=float)
    return tangent / n


def geodesic_angle(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    ua = np.asarray(a, dtype=float) / np.linalg.norm(a)
    ub = np.asarray(b, dtype=float) / np.linalg.norm(b)
    return float(np.arccos(np.clip(np.dot(ua, ub), -1.0, 1.0)))


@dataclass
class Node:
    position: npt.ArrayLike  # shape (3,)
    region: int | None = None  # assigned region index
    stress: float | None = None  # current stress
    density: float | None = None  # density based on region
    velocity: npt.ArrayLike | None = None  # shape (2,)
    friction: float | None = None  # calculated friction based on region properties


@dataclass
class Region:
    seed_node: int
    nodes: list[int]
    density: float  # initial density [0.01 to 1.0)
    velocity: npt.ArrayLike  # shape (2,)
    height: float  # initial height (-1.0 to 1.0)


class TectonicSimulation:
    nodes: list[Node]
    neighbors: dict[int, npt.ArrayLike]
    regions: list[Region]

    def __init__(
        self,
        nodes: list[Node],
        neighbors: dict[int, npt.ArrayLike],
        regions_count: int,
        seed: int | None = None,
        **configs,
    ):
        self.nodes = nodes
        self.neighbors = neighbors
        self.seed = seed if seed is not None else np.random.randint(0, 1_000_000)
        np.random.seed(self.seed)
        self.regions = self._generate_regions(regions_count, **configs)

        self._expand_regions(**configs)

    @classmethod
    def from_spheroid(
        cls, spheroid: Spheroid, regions_count: int, seed: int | None = None, **configs
    ) -> "TectonicSimulation":
        faces = spheroid.faces
        edges = spheroid.edges.astype(int)

        neighbors: dict[int, list[int]] = {i: [] for i in range(len(faces))}

        vertex_to_faces: dict[int, list[int]] = {}
        for face_index, face in enumerate(faces):
            for vertex_index in map(int, face):
                vertex_to_faces.setdefault(vertex_index, []).append(face_index)

        for start_vertex, end_vertex in edges:
            incident_faces = tuple(
                set(vertex_to_faces.get(int(start_vertex), ()))
                & set(vertex_to_faces.get(int(end_vertex), ()))
            )
            if len(incident_faces) == 2:
                neighbors[incident_faces[0]].append(incident_faces[1])
                neighbors[incident_faces[1]].append(incident_faces[0])

        neighbors = {
            node_index: np.array(sorted(set(node_neighbors)), dtype=int)
            for node_index, node_neighbors in neighbors.items()
        }
        # nodes initialized with zero stress, no region, no velocity
        # position is the face centroid
        nodes: list[Node] = []
        for face_index in neighbors.keys():
            face = faces[face_index]
            face_vertices = spheroid.vertices[np.array(face, dtype=int)]
            centroid = np.mean(face_vertices, axis=0)
            nodes.append(Node(position=centroid))

        return cls(
            nodes=nodes,
            neighbors=neighbors,
            regions_count=regions_count,
            seed=seed,
            **configs,
        )

    def _get_neighbors_within_radius(self, node_index: int, radius: int) -> list[int]:
        if radius <= 0:
            return []
        visited = {node_index}
        to_visit = {*self.neighbors[node_index]}
        for _ in range(radius):
            for current in to_visit.copy():
                visited.add(current)
                to_visit.discard(current)
                to_visit.update(
                    [
                        next_to_check
                        for next_to_check in self.neighbors[current]
                        if next_to_check not in visited
                    ]
                )
        visited.discard(node_index)
        return list(visited)

    def _node_traversal_distance(self, start_index: int, target_index: int) -> int:
        # use a* to find shortest path in graph
        if start_index == target_index:
            return 0
        visited = set()
        pq = PriorityQueue()
        pq.put((0, start_index, 0))  # (priority, node_index, distance)
        while not pq.empty():
            _, current_index, distance = pq.get()
            if current_index == target_index:
                return distance
            if current_index in visited:
                continue
            visited.add(current_index)
            for neighbor in self.neighbors[current_index]:
                if neighbor not in visited:
                    pq.put((distance + 1, neighbor, distance + 1))
        return float("inf")  # target not reachable

    def _generate_regions(self, regions_count: int, **configs) -> list[Region]:
        density_range = configs.get("density_range", (0.01, 1.0))
        velocity_range = configs.get("velocity_range", (-1.0, 1.0))
        height_range = configs.get("height_range", (-1.0, 1.0))
        seed_node_indices = np.random.choice(
            len(self.nodes), size=regions_count, replace=False
        ).tolist()
        regions: list[Region] = []
        for region_index, node_index in enumerate(seed_node_indices):
            node = self.nodes[node_index]
            node.region = region_index
            region = Region(
                seed_node=node_index,
                nodes=[node_index],
                density=np.random.uniform(*density_range),
                velocity=np.random.uniform(*velocity_range, size=2),
                height=np.random.uniform(*height_range),
            )
            node.velocity = region.velocity
            node.density = region.density
            regions.append(region)
        return regions

    def _expand_regions(self, **configs):
        velocity_decay = configs.get("velocity_decay", 0.1)
        density_decay = configs.get("density_decay", 0.1)
        while any(node.region is None for node in self.nodes):
            expandable_regions = [
                (region_index, region)
                for region_index, region in enumerate(self.regions)
                if any(
                    self.nodes[neighbor_index].region is None
                    for node_index in region.nodes
                    for neighbor_index in self.neighbors[node_index]
                )
            ]
            if not expandable_regions:
                print("No more expandable regions")
                if any(node.region is None for node in self.nodes):
                    raise RuntimeError("Some nodes could not be assigned to a region")
                break
            region_index, region = expandable_regions[
                np.random.choice(len(expandable_regions))
            ]
            border_nodes = [
                neighbor_index
                for node_index in region.nodes
                for neighbor_index in self.neighbors[node_index]
                if self.nodes[neighbor_index].region is None
            ]
            expansion_node_index = np.random.choice(border_nodes)
            distance = self._node_traversal_distance(
                expansion_node_index, region.seed_node
            )
            if distance == float("inf"):
                raise RuntimeError("Graph traversal failed")
            expansion_node = self.nodes[expansion_node_index]
            expansion_node.region = region_index
            expansion_node.velocity = (
                max(min(1.0 - distance * velocity_decay, 1.0), 0.01) * region.velocity
            )

            expansion_node.density = (
                max(min(1.0 - distance * density_decay, 1.0), 0.01) * region.density
            )
            region.nodes.append(expansion_node_index)

    def _initialize_friction(self, **config):
        contributing_neighbor_radius = config.get("contributing_neighbor_radius", 3)
        base_friction = config.get("base_friction", 1.0)
        position_weight = config.get("position_weight", 1.5)
        velocity_weight = config.get("velocity_weight", 0.2)
        density_weight = config.get("density_weight", 1.0)

        for node_index, node in enumerate(self.nodes):
            node_density = float(node.density) * density_weight
            node_point = np.asarray(node.position, dtype=float)
            node_vel3 = tangent_vector_from_lonlat_rates(
                node_point, np.asarray(node.velocity, dtype=float)
            )

            neighbor_indices = self._get_neighbors_within_radius(
                node_index, contributing_neighbor_radius
            )

            terms = []
            for ni in neighbor_indices:
                nb = self.nodes[ni]
                nb_density = float(nb.density) * density_weight
                nb_point = np.asarray(nb.position, dtype=float)
                nb_vel3 = tangent_vector_from_lonlat_rates(
                    nb_point, np.asarray(nb.velocity, dtype=float)
                )
                nb_vel3_at_node = parallel_transport_tangent_vector(
                    nb_point, node_point, nb_vel3
                )

                position_term = position_weight * geodesic_angle(node_point, nb_point)
                velocity_term = velocity_weight * np.linalg.norm(
                    node_vel3 - nb_vel3_at_node
                )
                friction_value = (base_friction + position_term + velocity_term) / (
                    (node_density + nb_density) / 2.0
                )
                terms.append(friction_value)

            node.friction = float(np.mean(terms)) if terms else base_friction

    def _initialize_stress(self, **config):
        velocity_amplification = config.get("velocity_amplification", 1.0)
        friction_dampening = config.get("friction_dampening", 1.0)

        for node_index, node in enumerate(self.nodes):
            node_point = np.asarray(node.position, dtype=float)
            node_vel3 = tangent_vector_from_lonlat_rates(
                node_point, np.asarray(node.velocity, dtype=float)
            )
            node_friction = float(node.friction)
            if node.stress is None:
                node.stress = 0.0

            for neighbor_index in self.neighbors[node_index]:
                neighbor = self.nodes[neighbor_index]
                neighbor_point = np.asarray(neighbor.position, dtype=float)
                neighbor_vel3 = tangent_vector_from_lonlat_rates(
                    neighbor_point, np.asarray(neighbor.velocity, dtype=float)
                )
                neighbor_vel3_at_node = parallel_transport_tangent_vector(
                    neighbor_point, node_point, neighbor_vel3
                )

                slip_dir = geodesic_unit_direction(node_point, neighbor_point)
                if not np.any(slip_dir):
                    continue

                rel = node_vel3 - neighbor_vel3_at_node
                approach = float(np.dot(rel, slip_dir))  # >0 together, <0 apart

                amp = (
                    1.0
                    + velocity_amplification
                    * (np.linalg.norm(node_vel3) + np.linalg.norm(neighbor_vel3))
                    / 2.0
                )
                denom = 1.0 + friction_dampening * (
                    node_friction + float(neighbor.friction)
                )

                node.stress += approach * amp / denom

    def setup_simulation(self, **config):
        """Initialize friction and stress for all nodes.

        Args:
            config: Configuration parameters for friction and stress initialization.
                - velocity_amplification: Amplification factor for velocity. Defaults to 1.0.
                - friction_dampening: Dampening factor for friction. Defaults to 1.0.
                - contributing_neighbor_radius: Radius of neighbors contributing to friction. Defaults to 3.
                - base_friction: Base friction value. Defaults to 1.0.
                - position_weight: Weight for position difference in friction calculation. Defaults to 1.5.
                - velocity_weight: Weight for velocity difference in friction calculation. Defaults to 0.2.
                - density_weight: Weight for density in friction calculation. Defaults to 1.0.
        """
        self._initialize_friction(**config)
        self._initialize_stress(**config)

    def step(self, **config):
        thresholds = config.get("stress_thresholds", np.arange(0.5, 5.5, 0.5))
        stress_thresholds = np.asarray(thresholds, dtype=float)
        stress_propagation_radius_per_threshold = config.get(
            "stress_propagation_radius_per_threshold", 1
        )
        stress_distribution_factor = config.get("stress_distribution_factor", 0.5)
        stable = True
        for node_index, node in enumerate(self.nodes):
            # check how many thresholds are exceeded
            node_stress = node.stress  # will never be None
            current_threshold = 0
            for threshold in stress_thresholds:
                if node_stress >= threshold:
                    current_threshold += 1
            if current_threshold == 0:
                continue
            stable = False
            # propagate stress to neighbors within radius
            propagation_radius = (
                current_threshold * stress_propagation_radius_per_threshold
            )
            neighbors_indices = self._get_neighbors_within_radius(
                node_index, propagation_radius
            )
            # stress distribution is reduced by graph distance and scaled by distribution factor

            stress_contribution = node_stress * stress_distribution_factor
            node.stress -= stress_contribution
            stress_per_neighbor = stress_contribution / len(neighbors_indices)
            for neighbor_index in neighbors_indices:
                neighbor = self.nodes[neighbor_index]
                distance = self._node_traversal_distance(node_index, neighbor_index)
                if np.isinf(distance):
                    raise RuntimeError("Graph traversal failed")
                if distance > 0:
                    neighbor.stress += stress_per_neighbor / distance
        return stable

    def run(self, cycles: int, **config):
        """Run the simulation for a number of cycles.

        Args:
            cycles (int): Maximum number of cycles to run.
            config: Configuration parameters for each simulation step.
                - stress_thresholds: callable returning array of stress thresholds. Defaults to [0.5, 1.0, ..., 5.0].
                - stress_propagation_radius_per_threshold: Radius of stress propagation per exceeded threshold. Defaults to 1.
                - stress_distribution_factor: Scale factor for stress distribution. Defaults to 0.5.
        """
        for _ in range(cycles):
            if self.step(**config):
                break

    def apply_to_spheroid(
        self, spheroid: Spheroid, stress_scalar: float = 1.0
    ) -> Spheroid:
        faces_stress = np.asarray(
            [node.stress + self.regions[node.region].height for node in self.nodes],
            dtype=float,
        )

        vertex_to_faces: dict[int, list[int]] = {}
        for face_index, face in enumerate(spheroid.faces):
            for vertex_index in map(int, face):
                vertex_to_faces.setdefault(vertex_index, []).append(face_index)

        for vertex_index, face_indices in vertex_to_faces.items():
            average_stress = float(
                np.mean(faces_stress[np.asarray(face_indices, dtype=int)])
            )
            spheroid.vectors[vertex_index][1] += np.exp(stress_scalar * average_stress)

        return spheroid
