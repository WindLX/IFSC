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

    results = {"average_throughput": [], "uav_num": [], "max_uav_num": []}
    for max_uav_num in [15, 20, 25]:
        uav_swarm.max_uav_num = max_uav_num
        for uav_num in tqdm(range(50, 251, 50), desc=f"max_uav_num={max_uav_num}"):
            uav_swarm.uav_num = uav_num
            uav_swarm.reset()
            uav_swarm.update()
            results["average_throughput"].append(uav_swarm.average_throughtput / 1e6)
            results["uav_num"].append(uav_num)
            results["max_uav_num"].append(max_uav_num)

    df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="uav_num", y="average_throughput", hue="max_uav_num", data=df)
    plt.xlabel("Number of UAVs")
    plt.ylabel("Average Throughput (Mbit)")
    plt.title("Average Throughput vs Number of UAVs for Different max_uav_num")
    plt.legend(title="max_uav_num")
    plt.savefig("image/average_throughput_uav_num_max_uav_num.png")
    plt.show()
