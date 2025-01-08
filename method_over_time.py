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

    methods = ["IFSC", "FSC", "GMM"]
    results = {
        method: {"average_throughput": [], "cluster_numbers": []} for method in methods
    }

    for velocity in methods:
        uav_swarm.method = velocity
        uav_swarm.reset()
        for _ in tqdm(
            range(0, simulation_time, delta_t), desc=f"Simulation Progress ({velocity})"
        ):
            uav_swarm.update()
            results[velocity]["average_throughput"].append(
                uav_swarm.average_throughtput / 1e6
            )

            results[velocity]["cluster_numbers"].append(uav_swarm.cluster_numbers)

    df_throughput = pd.DataFrame(
        {method: results[method]["average_throughput"] for method in methods}
    )
    df_throughput["time"] = range(0, simulation_time, delta_t)

    df_clusters = pd.DataFrame(
        {method: results[method]["cluster_numbers"] for method in methods}
    )
    df_clusters["time"] = range(0, simulation_time, delta_t)

    plt.figure(figsize=(12, 6))
    for velocity in methods:
        sns.lineplot(x="time", y=velocity, data=df_throughput, label=velocity)
    plt.title("Average Throughput Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Average Throughput (Mbit)")
    plt.grid(True)
    plt.legend()
    plt.savefig("image/throughput_time_method.png")
    plt.show()

    plt.figure(figsize=(12, 6))
    for velocity in methods:
        sns.lineplot(x="time", y=velocity, data=df_clusters, label=velocity)
    plt.title("Cluster Numbers Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Cluster Numbers")
    plt.grid(True)
    plt.legend()
    plt.savefig("image/cluster_number_uav_num_method.png")
    plt.show()
