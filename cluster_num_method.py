import numpy as np
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import *


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

    results = {"cluster_size": [], "method": []}
    fixed_uav_num = 100
    uav_swarm.uav_num = fixed_uav_num

    for method in ["IFSC", "FSC", "GMM"]:
        uav_swarm.method = method
        uav_swarm.reset()
        uav_swarm.update()
        cluster_sizes = [len(cluster.members) for cluster in uav_swarm.uav_clusters]
        for size in cluster_sizes:
            results["cluster_size"].append(size)
            results["method"].append(method)

    df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="method", y="cluster_size", data=df, palette="Set2")
    plt.xlabel("Method")
    plt.ylabel("Cluster Size")
    plt.title("Cluster Size Distribution for Different Methods")
    plt.savefig("image/cluster_size_distribution_method.png")
    plt.show()
