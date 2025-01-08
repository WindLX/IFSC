import numpy as np
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import *


if __name__ == "__main__":
    position_bound = (np.array([0, 0, 100]), np.array([2000, 2000, 150]))
    uav_num = 100
    method = 15
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
        method,
        position_bound,
        velocity_mean,
        velocity_std,
        velocity_bound,
        delta_t,
        signal_config,
        seed=seed,
    )

    uav_swarm.reset()

    results = {"time": [], "average_throughput": [], "velocity": []}
    uav_num = 100
    for velocity in [10, 20, 30]:
        uav_swarm.velocity_mean = velocity
        uav_swarm.uav_num = uav_num
        uav_swarm.reset()
        for t in tqdm(range(0, simulation_time, delta_t), desc=f"velocity={velocity}"):
            uav_swarm.update()
            results["time"].append(t)
            results["average_throughput"].append(uav_swarm.average_throughtput / 1e6)
            results["velocity"].append(velocity)

    df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    sns.lineplot(x="time", y="average_throughput", hue="velocity", data=df)
    plt.xlabel("Time (s)")
    plt.ylabel("Average Throughput (Mbit)")
    plt.title("Average Throughput vs Time for Different Velocity")
    plt.legend(title="velocity")
    plt.savefig("image/average_throughput_vs_time_for_different_velocity.png")
    plt.show()
