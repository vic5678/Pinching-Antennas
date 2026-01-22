import itertools
import numpy as np
import json
import os
import time
from QFP_gurobi import solve_qf01p_unit_d 
from System_model_setup import generate_normalized_pinch_positions, get_antenna_parameters, beta_calc
import torch 

def format_complex_for_json(B_sample):
    """Converts a complex vector to list of cleaned strings."""
    return [[f"({val.real}+{val.imag}j)" if val.imag >= 0 else f"({val.real}{val.imag}j)"] for val in B_sample]

def generate_augmented_dataset_with_gurobi(n_antennas, n_samples, n_noisy_per_user, save_path, time_limit=3600, OutputFlag=0, sigma_value=0.3):
    parameters = get_antenna_parameters(no_antennas=n_antennas, no_users=1)
    pinch_positions = generate_normalized_pinch_positions(parameters)
    dataset = []
    t0 = time.time()

    for i in range(n_samples):
        # === Generate clean user position ===
        #user_position = np.random.uniform(low=-parameters["SQUARE_SIDE"], high=parameters["SQUARE_SIDE"], size=(1, 3))
        #user_position[:, 2] = np.random.uniform(low=0, high=1)
        user_position = np.random.uniform(low=-parameters["SQUARE_SIDE"], high=parameters["SQUARE_SIDE"], size=(1, 3))
        user_position[:, 2] = np.random.uniform(low=0, high=1)
 
 

        for noise_idx in range(n_noisy_per_user):
            # === Add noise to clean position ===
            user_pos = torch.from_numpy(user_position).unsqueeze(1).to(torch.float32)
            noise = torch.randn_like(user_pos) * sigma_value
            noisy_user_pos = user_pos + noise

            # === Calculate clean and noisy B ===
            B_clean = beta_calc(n_antennas, pinch_positions, user_position, parameters["LAMBDA"], parameters["LAMBDA_G"], parameters["PSI_0"], batch_size=1)
            B_noisy = beta_calc(n_antennas, pinch_positions, noisy_user_pos.squeeze(1).cpu().numpy(), parameters["LAMBDA"], parameters["LAMBDA_G"], parameters["PSI_0"], batch_size=1)

            B_sample_clean = B_clean[0].flatten()
            B_sample_noisy = B_noisy[0].flatten()

            # === Solve with Gurobi ===
            Q = np.real(np.outer(B_sample_clean, B_sample_clean.conj()))
            x_sol, num, denom, obj = solve_qf01p_unit_d(Q, timeLimit=time_limit, Output_Flag=OutputFlag)

            obj = obj*parameters["ETA"] *parameters["P"]/ parameters["SIGMA2"]
            rate = float(np.log2(1 + obj))

            sample = {
                "user_position": noisy_user_pos[0].tolist(),
                "B": format_complex_for_json(B_sample_clean),
                "B_noisy": format_complex_for_json(B_sample_noisy),
                "optimal_rate": rate,
                "SNR": obj,
                "a_opt": x_sol.astype(int).tolist()
            }
            dataset.append(sample)

            print(f"✔ Instance {i+1}/{n_samples}, Noisy sample {noise_idx+1}/{n_noisy_per_user} completed")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\n✅ Dataset saved successfully at {save_path}")
    print(f"⏱ Total time: {time.time() - t0:.2f} seconds")


if __name__ == "__main__":
    n_antennas = 45
    n_samples = 1 # how many user positions
    n_noisy_per_user = 1 # how many noisy samples per user position
    timeLimit = 10
    goal = "test"
    sigma = 0
    save_path = f"data/{goal}/dataset_{n_samples}samples{n_antennas}ant_OOD.json"
    #save_path = f"data/{goal}/dataset_{n_samples}_{n_antennas}ant_sigma{sigma}.json"
    generate_augmented_dataset_with_gurobi(
        n_antennas=n_antennas,
        n_samples=n_samples,
        n_noisy_per_user=n_noisy_per_user,
        save_path=save_path,
        time_limit=timeLimit,
        OutputFlag=0,
        sigma_value=sigma
    )
