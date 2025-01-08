import numpy as np
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import *


def plot_clusters(uav_swarm, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for cluster_id, cluster in enumerate(uav_swarm.uav_clusters):
        positions = np.array([uav.position for uav in cluster.members])
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            label=f"Cluster {cluster_id}",
        )
        cluster_head = cluster.ch.position
        ax.scatter(
            cluster_head[0],
            cluster_head[1],
            cluster_head[2],
            marker="*",
            s=200,
            c=ax._get_lines.get_next_color(),
        )
    ax.set_title(title)
    ax.legend()
    plt.savefig(f"image/{title}.png")
    plt.show()


if __name__ == "__main__":
    position_bound = (np.array([0, 0, 100]), np.array([2000, 2000, 150]))
    uav_num = 100
    velocity = 15
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
        prediction_horizon=1,
        discount_factor=0.1,
    )

    uav_swarm = UAVSwarm(
        uav_num,
        velocity,
        position_bound,
        velocity_mean,
        velocity_std,
        velocity_bound,
        delta_t,
        signal_config,
        seed=seed,
    )

    uav_swarm.reset()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        uav_swarm.positions[:, 0],
        uav_swarm.positions[:, 1],
        uav_swarm.positions[:, 2],
    )
    ax.set_title("Initial UAV Positions")
    plt.savefig("image/initial_uav_positions.png")
    plt.show()

    for velocity in [15, 20, 25]:
        uav_swarm.max_uav_num = velocity
        uav_swarm.reset()
        plot_clusters(uav_swarm, f"UAV Clusters (max_uav_num={velocity})")
