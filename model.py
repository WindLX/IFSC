from dataclasses import dataclass

import numpy as np
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


@dataclass
class SignalConfig:
    # P
    transmit_power: float
    # h
    channel_gain: float
    # alpha
    path_loss_exponent: float
    # sigma^2
    noise_power: float
    # B
    bandwidth: float
    # gamma
    sinr_threshold: float
    # varepsilon_fs
    amplifier_consumption: float
    # E_elec
    transmitter_consumption: float
    # l
    packet_length: float
    # theta
    adjustable_factor: float
    # H
    prediction_horizon: int
    # nu
    discount_factor: float


def dBm_to_watt(P_dBm):
    return 10 ** ((P_dBm - 30) / 10)


def dB_to_watt(P_dB):
    return 10 ** (P_dB / 10)


class UAV:
    def __init__(
        self,
        id: int,
        position_bound: tuple[np.ndarray, np.ndarray],
        velocity_mean: float,
        velocity_std: float,
        velocity_bound: tuple[float, float],
        delta_t: float,
        signal_config: SignalConfig,
        seed: int = None,
    ):
        self.id = id

        self.velocity_mean = velocity_mean
        self.velocity_std = velocity_std
        self.velocity_bound = velocity_bound
        self.delta_t = delta_t
        self.position_bound = position_bound

        self.signal_config = signal_config

        self.position_history = []
        self.velocity_history = []

        if seed is not None:
            np.random.seed(seed)

    def randomize_velocity(self) -> np.ndarray:
        velocity_norm = np.abs(np.random.normal(self.velocity_mean, self.velocity_std))
        velocity_norm = np.clip(
            velocity_norm, self.velocity_bound[0], self.velocity_bound[1]
        )
        velocity_varphi = np.random.uniform(0, 2 * np.pi)
        velocity_theta = np.random.uniform(0, np.pi)

        velocity = np.array(
            [
                velocity_norm * np.sin(velocity_theta) * np.cos(velocity_varphi),
                velocity_norm * np.sin(velocity_theta) * np.sin(velocity_varphi),
                velocity_norm * np.cos(velocity_theta),
            ]
        )
        return velocity

    def update_velocity(self) -> np.ndarray:
        # velocity = self.randomize_velocity()
        velocity = self.velocity

        predictive_position = self.position + velocity * self.delta_t
        for i in range(len(self.position)):
            if (
                not self.position_bound[0][i]
                <= predictive_position[i]
                <= self.position_bound[1][i]
            ):
                velocity[i] = -velocity[i]

        self.velocity = velocity
        return self.velocity

    def update_position(self) -> np.ndarray:
        self.position = self.position + self.velocity * self.delta_t
        self.position_history.append(self.position)
        self.velocity_history.append(self.velocity)
        return self.position

    def reset(self):
        self.position = np.random.uniform(
            self.position_bound[0], self.position_bound[1]
        )
        self.velocity = self.randomize_velocity()
        self.position_history = [self.position]
        self.velocity_history = [self.velocity]

    def update(self):
        self.update_velocity()
        self.update_position()

    def distance(self, other: "UAV") -> float:
        return np.linalg.norm(self.position - other.position, ord=2)

    def last_distance(self, other: "UAV") -> float:
        return np.linalg.norm(
            self.position_history[-2] - other.position_history[-2], ord=2
        )

    def delta_distance(self, other: "UAV") -> float:
        return self.distance(other) - self.last_distance(other)

    def relative_speed(self, other: "UAV") -> float:
        upper = np.dot(
            (self.position - other.position), (self.velocity - other.velocity)
        )
        lower = self.distance(other)
        return np.abs(upper / lower)

    def connection_time(self, other: "UAV") -> float:
        if len(self.position_history) < 2:
            if self.distance(other) < self.max_communication_distance:
                return self.delta_t
            else:
                return 0

        if (
            self.distance(other) >= self.max_communication_distance
            or self.last_distance(other) >= self.max_communication_distance
        ):
            return 0

        temp_connection_time = 0
        delta_distance = self.delta_distance(other)
        if delta_distance < 0:
            temp_connection_time = (
                self.max_communication_distance + self.distance(other)
            ) / self.relative_speed(other)
        elif delta_distance > 0:
            temp_connection_time = (
                self.max_communication_distance - self.distance(other)
            ) / self.relative_speed(other)
        else:
            temp_connection_time = np.inf

        if temp_connection_time < self.delta_t:
            return temp_connection_time
        else:
            return self.delta_t

    @staticmethod
    def sinr(distance: float, signal_config: SignalConfig) -> float:
        sinr = (
            signal_config.transmit_power
            * signal_config.channel_gain
            * distance**-signal_config.path_loss_exponent
            / signal_config.noise_power
        )
        return sinr

    def current_sinr(self, other: "UAV") -> float:
        distance = self.distance(other)
        sinr = self.sinr(distance, self.signal_config)
        return sinr

    def last_sinr(self, other: "UAV") -> float:
        distance = self.last_distance(other)
        sinr = self.sinr(distance, self.signal_config)
        return sinr

    @staticmethod
    def capacity(sinr: float, signal_config: SignalConfig) -> float:
        capacity = signal_config.bandwidth * np.log(1 + sinr)
        return capacity

    def current_capacity(self, other: "UAV") -> float:
        sinr = self.current_sinr(other)
        capacity = self.capacity(sinr, self.signal_config)
        return capacity

    def last_capacity(self, other: "UAV") -> float:
        sinr = self.last_sinr(other)
        capacity = self.capacity(sinr, self.signal_config)
        return capacity

    def throughput(self, other: "UAV") -> float:
        connection_time = self.connection_time(other)
        capacity = self.current_capacity(other)
        if len(self.position_history) < 2:
            last_capacity = capacity
        else:
            last_capacity = self.last_capacity(other)

        throughput = (last_capacity + capacity) / 2 * connection_time
        return throughput

    def link_efficiency(self, other: "UAV") -> float:
        distance = self.distance(other)
        connect_time = self.connection_time(other)
        beta_s = connect_time / self.delta_t

        E_upper = (
            self.signal_config.packet_length
            * self.signal_config.transmitter_consumption
            + self.signal_config.packet_length
            * self.signal_config.amplifier_consumption
            * distance**2
        )
        E_lower = (
            self.signal_config.packet_length
            * self.signal_config.transmitter_consumption
            + self.signal_config.packet_length
            * self.signal_config.amplifier_consumption
            * self.max_communication_distance**2
        )

        beta_e = 1 - E_upper / E_lower

        beta = (
            self.signal_config.adjustable_factor * beta_s
            + (1 - self.signal_config.adjustable_factor) * beta_e
        )

        Q = beta * self.current_capacity(other)
        Q /= 1e6

        return Q

    def future_link_efficiency(self, other: "UAV") -> float:
        nu = self.signal_config.discount_factor
        H = self.signal_config.prediction_horizon

        future_Q = 0
        original_position = self.position.copy()
        original_velocity = self.velocity.copy()
        other_original_position = other.position.copy()
        other_original_velocity = other.velocity.copy()

        for h in range(1, H + 1):
            self.update_velocity()
            self.update_position()
            other.update_velocity()
            other.update_position()

            distance = self.distance(other)
            connect_time = self.connection_time(other)
            beta_s = connect_time / self.delta_t

            E_upper = (
                self.signal_config.packet_length
                * self.signal_config.transmitter_consumption
                + self.signal_config.packet_length
                * self.signal_config.amplifier_consumption
                * distance**2
            )
            E_lower = (
                self.signal_config.packet_length
                * self.signal_config.transmitter_consumption
                + self.signal_config.packet_length
                * self.signal_config.amplifier_consumption
                * self.max_communication_distance**2
            )

            beta_e = 1 - E_upper / E_lower

            beta = (
                self.signal_config.adjustable_factor * beta_s
                + (1 - self.signal_config.adjustable_factor) * beta_e
            )

            Q = beta * self.current_capacity(other)
            Q /= 1e6

            future_Q += nu**h * Q

        self.position = original_position
        self.velocity = original_velocity
        other.position = other_original_position
        other.velocity = other_original_velocity

        return future_Q

    @property
    def max_communication_distance(self) -> float:
        varepsilon = (
            self.signal_config.transmit_power
            * self.signal_config.channel_gain
            / (self.signal_config.sinr_threshold * self.signal_config.noise_power)
        ) ** (1 / self.signal_config.path_loss_exponent)
        return varepsilon

    def __str__(self):
        return f"UAV: id={self.id}, position={self.position}, velocity={self.velocity}"

    def __repr__(self):
        return f"UAV: id={self.id}, position={self.position}, velocity={self.velocity}"


class UAVCluster:
    def __init__(self, ch: UAV, members: list[UAV]):
        self.ch = ch
        self.members = members
        self.members.remove(ch)

    @property
    def throughput(self) -> float:
        throughput = 0
        for member in self.members:
            throughput += self.ch.throughput(member)
        return throughput

    @property
    def member_numbers(self) -> int:
        return len(self.members)

    def add_member(self, member: UAV):
        self.members.append(member)

    def remove_member(self, member: UAV):
        self.members.remove(member)

    def move_member(self, member: UAV, other: "UAVCluster"):
        self.members.remove(member)
        other.add_member(member)

    def set_ch(self, ch: UAV):
        self.ch = ch

    def check_ch(self):
        for member in self.members:
            if self.ch.current_sinr(member) < self.ch.signal_config.sinr_threshold:
                return False

    def display(self):
        print(
            f"Cluster: CH={self.ch.id}, Members={[member.id for member in self.members]}, Throughput={self.throughput}"
        )


class UAVSwarm:
    def __init__(
        self,
        uav_num: int,
        max_uav_num: int,
        position_bound: tuple[np.ndarray, np.ndarray],
        velocity_mean: float,
        velocity_std: float,
        velocity_bound: tuple[float, float],
        delta_t: float,
        signal_config: SignalConfig,
        method: str = "IFSC",
        seed: int = None,
    ):
        self.uav_num = uav_num
        self.max_uav_num = max_uav_num
        self.max_swarm_num = uav_num // max_uav_num

        self.init_position_bound = position_bound
        self.velocity_mean = velocity_mean
        self.velocity_std = velocity_std
        self.delta_t = delta_t
        self.signal_config = signal_config

        self.method = method

        if seed is not None:
            np.random.seed(seed)

        self.uavs = [
            UAV(
                id,
                position_bound,
                velocity_mean,
                velocity_std,
                velocity_bound,
                delta_t,
                signal_config,
                seed,
            )
            for id in range(uav_num)
        ]

        self.uav_clusters = []

    def reset(self):
        for uav in self.uavs:
            uav.reset()
        self.reset_clusters()

    def update(self):
        for uav in self.uavs:
            uav.update()
        self.update_clusters()

    @property
    def positions(self):
        return np.array([uav.position for uav in self.uavs])

    @property
    def cluster_heads(self):
        return [cluster.ch.id for cluster in self.uav_clusters]

    def select_cluster_head(self, cluster):
        cand = []
        for i in cluster:
            count = 0
            for j in cluster:
                if i != j and i.current_sinr(j) >= self.signal_config.sinr_threshold:
                    count += 1
            if count == len(cluster) - 1:
                cand.append(i)

        if not cand:
            return None

        ch = max(
            cand, key=lambda k: sum(k.link_efficiency(j) for j in cluster if j != k)
        )
        return ch

    def improved_select_cluster_head(self, cluster):
        cand = []
        future_validity = []

        for i in cluster:
            count = 0
            for j in cluster:
                if i != j and i.current_sinr(j) >= self.signal_config.sinr_threshold:
                    count += 1
            if count == len(cluster) - 1:
                cand.append(i)

        for i in cand:
            future_Q = i.future_link_efficiency(cluster[0], nu=0.9, H=10)
            future_validity.append(future_Q)

        sorted_cand = [x for _, x in sorted(zip(future_validity, cand), reverse=True)]
        k = sorted_cand[0] if sorted_cand else None

        return k, cand

    def call_method(self):
        if self.method == "IFSC":
            self.improved_fission_spectral_clustering()
        elif self.method == "FSC":
            self.fission_spectral_clustering()
        elif self.method == "GMM":
            self.gmm()
        elif self.method == "AP":
            self.affinity_propagation()
        elif self.method == "AC":
            self.agglomerative_clustering()

    def fission_spectral_clustering(self):
        N = len(self.uavs)
        W = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                if i != j:
                    if (
                        self.uavs[i].current_sinr(self.uavs[j])
                        >= self.signal_config.sinr_threshold
                    ):
                        W[i, j] = self.uavs[i].link_efficiency(self.uavs[j])
                    else:
                        W[i, j] = 0

        D = np.diag(W.sum(axis=1))
        D_inv_sqrt = np.linalg.inv(np.sqrt(D))
        L = np.eye(N) - D_inv_sqrt @ W @ D_inv_sqrt

        K = int(np.ceil(N / self.max_uav_num))
        eigvals, eigvecs = np.linalg.eigh(L)
        F = eigvecs[:, :K]
        F = F / np.linalg.norm(F, axis=1, keepdims=True)

        kmeans = KMeans(n_clusters=K)
        labels = kmeans.fit_predict(F)

        self.uav_clusters = [
            UAVCluster(
                self.select_cluster_head(
                    [self.uavs[i] for i in range(len(self.uavs)) if labels[i] == j]
                ),
                [self.uavs[i] for i in range(len(self.uavs)) if labels[i] == j],
            )
            for j in range(K)
        ]

        results = self.uav_clusters.copy()
        while any(len(cluster.members) > self.max_uav_num for cluster in results):
            new_results = []
            for cluster in results:
                if len(cluster.members) > self.max_uav_num:
                    K_prime = int(np.ceil(len(cluster.members) / self.max_uav_num))
                    positions = np.array([uav.position for uav in cluster.members])
                    kmeans = KMeans(n_clusters=K_prime)
                    sub_labels = kmeans.fit_predict(positions)
                    sub_clusters = [
                        UAVCluster(
                            self.select_cluster_head(
                                [
                                    cluster.members[i]
                                    for i in range(len(cluster.members))
                                    if sub_labels[i] == j
                                ]
                            ),
                            [
                                cluster.members[i]
                                for i in range(len(cluster.members))
                                if sub_labels[i] == j
                            ],
                        )
                        for j in range(K_prime)
                    ]
                    new_results.extend(sub_clusters)
                else:
                    new_results.append(cluster)
            results = new_results

        final_results = []
        for cluster in results:
            while not any(self.select_cluster_head([uav]) for uav in cluster.members):
                K_prime = 2
                positions = np.array([uav.position for uav in cluster.members])
                kmeans = KMeans(n_clusters=K_prime)
                sub_labels = kmeans.fit_predict(positions)
                sub_clusters = [
                    UAVCluster(
                        self.select_cluster_head(
                            [
                                cluster.members[i]
                                for i in range(len(cluster.members))
                                if sub_labels[i] == j
                            ]
                        ),
                        [
                            cluster.members[i]
                            for i in range(len(cluster.members))
                            if sub_labels[i] == j
                        ],
                    )
                    for j in range(K_prime)
                ]
                results.remove(cluster)
                results.extend(sub_clusters)
            final_results.append(cluster)

        self.uav_clusters = final_results

    def improved_fission_spectral_clustering(self):
        N = len(self.uavs)
        W = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                if i != j:
                    if (
                        self.uavs[i].current_sinr(self.uavs[j])
                        >= self.signal_config.sinr_threshold
                    ):
                        W[i, j] = self.uavs[i].link_efficiency(self.uavs[j])
                    else:
                        W[i, j] = 0

        D = np.diag(W.sum(axis=1))
        D_inv_sqrt = np.linalg.inv(np.sqrt(D))
        L = np.eye(N) - D_inv_sqrt @ W @ D_inv_sqrt

        K = int(np.ceil(N / self.max_uav_num))
        eigvals, eigvecs = np.linalg.eigh(L)
        F = eigvecs[:, :K]
        F = F / np.linalg.norm(F, axis=1, keepdims=True)

        kmeans = KMeans(n_clusters=K)
        labels = kmeans.fit_predict(F)

        self.uav_clusters = [
            UAVCluster(
                self.select_cluster_head(
                    [self.uavs[i] for i in range(len(self.uavs)) if labels[i] == j]
                ),
                [self.uavs[i] for i in range(len(self.uavs)) if labels[i] == j],
            )
            for j in range(K)
        ]

        results = self.uav_clusters.copy()
        while any(len(cluster.members) > self.max_uav_num for cluster in results):
            new_results = []
            for cluster in results:
                if len(cluster.members) > self.max_uav_num:
                    K_prime = int(np.ceil(len(cluster.members) / self.max_uav_num))
                    positions = np.array([uav.position for uav in cluster.members])
                    kmeans = KMeans(n_clusters=K_prime)
                    sub_labels = kmeans.fit_predict(positions)
                    sub_clusters = [
                        UAVCluster(
                            self.select_cluster_head(
                                [
                                    cluster.members[i]
                                    for i in range(len(cluster.members))
                                    if sub_labels[i] == j
                                ]
                            ),
                            [
                                cluster.members[i]
                                for i in range(len(cluster.members))
                                if sub_labels[i] == j
                            ],
                        )
                        for j in range(K_prime)
                    ]
                    new_results.extend(sub_clusters)
                else:
                    new_results.append(cluster)
            results = new_results

        final_results = []
        for cluster in results:
            while not any(self.select_cluster_head([uav]) for uav in cluster.members):
                K_prime = 2
                positions = np.array([uav.position for uav in cluster.members])
                kmeans = KMeans(n_clusters=K_prime)
                sub_labels = kmeans.fit_predict(positions)
                sub_clusters = [
                    UAVCluster(
                        self.select_cluster_head(
                            [
                                cluster.members[i]
                                for i in range(len(cluster.members))
                                if sub_labels[i] == j
                            ]
                        ),
                        [
                            cluster.members[i]
                            for i in range(len(cluster.members))
                            if sub_labels[i] == j
                        ],
                    )
                    for j in range(K_prime)
                ]
                results.remove(cluster)
                results.extend(sub_clusters)
            final_results.append(cluster)

        self.uav_clusters = final_results

    def gmm(self):
        N = len(self.uavs)
        positions = np.array([uav.position for uav in self.uavs])
        gmm = GaussianMixture(n_components=self.max_swarm_num)
        labels = gmm.fit_predict(positions)

        self.uav_clusters = [
            UAVCluster(
                self.select_cluster_head(
                    [self.uavs[i] for i in range(len(self.uavs)) if labels[i] == j]
                ),
                [self.uavs[i] for i in range(len(self.uavs)) if labels[i] == j],
            )
            for j in range(self.max_swarm_num)
        ]

    def affinity_propagation(self):
        N = len(self.uavs)
        positions = np.array([uav.position for uav in self.uavs])
        ap = AffinityPropagation()
        labels = ap.fit_predict(positions)

        self.uav_clusters = [
            UAVCluster(
                self.select_cluster_head(
                    [self.uavs[i] for i in range(len(self.uavs)) if labels[i] == j]
                ),
                [self.uavs[i] for i in range(len(self.uavs)) if labels[i] == j],
            )
            for j in range(len(set(labels)))
        ]

    def agglomerative_clustering(self):
        N = len(self.uavs)
        positions = np.array([uav.position for uav in self.uavs])
        ac = AgglomerativeClustering(n_clusters=self.max_swarm_num)
        labels = ac.fit_predict(positions)

        self.uav_clusters = [
            UAVCluster(
                self.select_cluster_head(
                    [self.uavs[i] for i in range(len(self.uavs)) if labels[i] == j]
                ),
                [self.uavs[i] for i in range(len(self.uavs)) if labels[i] == j],
            )
            for j in range(self.max_swarm_num)
        ]

    def maintain_clusters(self):
        if self.throughput < self.threshold_throughput:
            self.call_method()
            return

        if self.method == "IFSC" or self.method == "FSC":
            for cluster in self.uav_clusters:
                for uav in cluster.members:
                    while (
                        uav.current_sinr(cluster.ch) < self.signal_config.sinr_threshold
                    ):
                        new_neighbor = [
                            other
                            for other in self.uavs
                            if other != uav
                            and uav.current_sinr(other)
                            >= self.signal_config.sinr_threshold
                        ]
                        if new_neighbor:
                            if self.method == "IFSC":
                                best_neighbor, _ = self.improved_select_cluster_head(
                                    new_neighbor
                                )
                            else:
                                best_neighbor = max(
                                    new_neighbor, key=lambda x: uav.link_efficiency(x)
                                )
                            cluster.add_member(best_neighbor)
                            self.uav_clusters[
                                self.uav_clusters.index([best_neighbor])
                            ].remove_member(best_neighbor)

        for cluster in self.uav_clusters:
            if len(cluster.members) > self.max_uav_num or not any(
                self.select_cluster_head([uav]) for uav in cluster.members
            ):
                self.call_method()

    def reset_clusters(self):
        self.uav_clusters = []

        self.call_method()

        self.threshold_throughput = self.throughput * 0.5

    def update_clusters(self):
        self.maintain_clusters()

    @property
    def throughput(self) -> float:
        throughput = 0
        for cluster in self.uav_clusters:
            throughput += cluster.throughput
        return throughput

    @property
    def average_throughtput(self) -> float:
        return self.throughput / self.uav_num

    @property
    def cluster_numbers(self) -> int:
        return len(self.uav_clusters)

    def display(self, id: int | None = None):
        if id is not None:
            print(self.uavs[id])
        else:
            for uav in self.uavs:
                print(uav)

    def display_clusters(self, id: int | None = None):
        if id is not None:
            self.uav_clusters[id].display()
        else:
            for cluster in self.uav_clusters:
                cluster.display()

    def display_connection_info(self, ids: tuple[int, int]):
        uav = self.uavs[ids[0]]
        other = self.uavs[ids[1]]
        print(f"[{uav.id}] to [{other.id}]:")
        print(f"  Distance: {uav.distance(other)}")
        print(f"  Connection Time: {uav.connection_time(other)}")
        print(f"  SINR: {uav.current_sinr(other)}")
        print(f"  Capacity: {uav.current_capacity(other)}")
        print(f"  Throughput: {uav.throughput(other)}")
        print(f"  Link Efficiency: {uav.link_efficiency(other)}")

    def display_cluster_connection_info(self, cluster_id: int):
        cluster = self.uav_clusters[cluster_id]
        printed_pairs = set()
        for uav in cluster.members:
            for other in cluster.members:
                if (
                    uav.id != other.id
                    and (uav.id, other.id) not in printed_pairs
                    and (other.id, uav.id) not in printed_pairs
                ):
                    printed_pairs.add((uav.id, other.id))
                    print(f"[{uav.id}] to [{other.id}]:")
                    print(f"  Distance: {uav.distance(other)}")
                    print(f"  Connection Time: {uav.connection_time(other)}")
                    print(f"  SINR: {uav.current_sinr(other)}")
                    print(f"  Capacity: {uav.current_capacity(other)}")
                    print(f"  Throughput: {uav.throughput(other)}")
                    print(f"  Link Efficiency: {uav.link_efficiency(other)}")


if __name__ == "__main__":
    position_bound = (np.array([0, 0, 100]), np.array([2000, 2000, 150]))
    uav_num = 2
    max_uav_num = 20
    velocity_mean = 20
    velocity_std = 5
    velocity_bound = (10, 30)
    delta_t = 10
    simulation_time = 300
    seed = 42

    signal_config = SignalConfig(
        transmit_power=dBm_to_watt(20),
        channel_gain=0.5,
        path_loss_exponent=4,
        noise_power=dBm_to_watt(-100),
        bandwidth=10e6,
        sinr_threshold=dBm_to_watt(0),
        amplifier_consumption=10e-12,
        transmitter_consumption=50e-9,
        packet_length=1,
        adjustable_factor=0.5,
        prediction_horizon=5,
        discount_factor=0.5,
    )

    uav_swarm = UAVSwarm(
        uav_num,
        max_uav_num,
        position_bound,
        velocity_mean,
        velocity_std,
        velocity_bound,
        delta_t,
        signal_config,
        seed,
    )

    methods = ["FSC"]
    results = {
        method: {"average_throughput": [], "cluster_numbers": []} for method in methods
    }

    for method in methods:
        uav_swarm.method = method
        uav_swarm.reset()
        for _ in tqdm(
            range(0, simulation_time, delta_t), desc=f"Simulation Progress ({method})"
        ):
            uav_swarm.update()
            results[method]["average_throughput"].append(
                uav_swarm.average_throughtput / 1e6
            )
            results[method]["cluster_numbers"].append(uav_swarm.cluster_numbers)
            for cl in uav_swarm.uav_clusters:
                cl.display()
                print()
