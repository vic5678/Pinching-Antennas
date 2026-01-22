import torch
from torch.distributions import Categorical
from policy_BCE import Policy
from policy_GNN_MLP import Policy_GNN_MLP
from torch_geometric.data import Batch , Data
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os
from System_model_setup import plot_antenna_selection, beta_calc, calculate_rates, generate_normalized_pinch_positions, get_antenna_parameters, preprocess_data, create_batch_graph, load_dataset, calculate_antenna_activation
from QFP_gurobi import solve_qf01p_unit_d

def format_complex_for_json(B_sample):
    """Converts a complex vector to list of cleaned strings."""
    return [[f"({val.real}+{val.imag}j)" if val.imag >= 0 else f"({val.real}{val.imag}j)"] for val in B_sample]

def test_model_with_gurobi(test_size, test_dataset_path, n_antennas_test, n_users, pinch_positions, parameters, dev, K=100, sigma=0.3):
    # Load test data
    user_positions_test, _, B_test_correct, _, optimal_rates_test, optimal_SNRs_test, a_opts_test = preprocess_data(
        dataset_path=test_dataset_path,
        total_samples=test_size,
        device=dev
    )

    antenna_activation = calculate_antenna_activation(a_opts_test)
    print("Antenna activation (without noise): ", antenna_activation)

    final_actions = []

    for i in range(test_size):
        user_pos = torch.from_numpy(user_positions_test[i]).unsqueeze(0).to(torch.float32).to(dev)  # shape: [1, 3]
        noisy_actions = []

        if i % 10 == 0:
            print("Sample: ", i)

        for k in range(K):
            # Apply Gaussian noise
            noise = torch.randn_like(user_pos) * sigma
            noisy_user_pos = user_pos + noise

            # Compute B for noisy user position
            B_sample = beta_calc(
                parameters["N_PINCHES"],
                pinch_positions,
                noisy_user_pos.cpu().numpy(),
                parameters["LAMBDA"],
                parameters["LAMBDA_G"],
                parameters["PSI_0"],
                batch_size=1
            )[0]

            B_sample = B_sample.flatten()
            Q = np.real(np.outer(B_sample, B_sample.conj()))

            # Solve with Gurobi
            x_sol, _, _, _ = solve_qf01p_unit_d(Q, timeLimit=2, Output_Flag=0)
            noisy_actions.append(x_sol.astype(int))

        # Average over K and threshold to get binary final action
        mean_action = np.mean(noisy_actions, axis=0)
        final_action = (mean_action > 0.5).astype(int)
        final_actions.append(final_action)

    # Convert to expected shape: [test_size, n_users, n_antennas]
    a_opts_gurobi = np.array(final_actions).reshape(test_size, n_users, n_antennas_test)

    # Evaluate SNRs and rates
    gurobi_rates_test, gurobi_SNRs = calculate_rates(
        B=B_test_correct,
        batch_size=test_size,
        a_opt=torch.tensor(a_opts_gurobi),
        parameters=parameters
    )

    # Compute average SNR ratio (Gurobi vs optimal)
    total_SNR_ratio = 0.0
    gurobi_SNRs = np.array(gurobi_SNRs)
    print("gurobi_SNRs", gurobi_SNRs)
    for i in range(test_size):
        SNR_gurobi = gurobi_SNRs[i]
        SNR_optimal = optimal_SNRs_test[i]
        total_SNR_ratio += SNR_gurobi / SNR_optimal

    avg_gurobi_SNR_ratio = total_SNR_ratio / test_size
    print(f"✅ Average Gurobi SNR Ratio (K={K} samples, sigma={sigma:.2f}): {avg_gurobi_SNR_ratio:.4f}")

    return avg_gurobi_SNR_ratio, a_opts_gurobi, user_positions_test, B_test_correct, gurobi_SNRs

def save_gurobi_eval_dataset(user_positions_test, B_test_correct, gurobi_SNRs_test, a_opts_gurobi, parameters, save_path):
    dataset = []
    for i in range(len(user_positions_test)):
        user_pos = user_positions_test[i].tolist()
        B_clean = B_test_correct[i].flatten()
        snr = float(gurobi_SNRs_test[i])
        a_opt = a_opts_gurobi[i][0].tolist()
        rate = float(np.log2(1 + snr))
        sample = {
            "user_position": user_pos,
            "B": format_complex_for_json(B_clean),
            "B_noisy": format_complex_for_json(B_clean),
            "optimal_rate": rate,
            "SNR": snr,
            "a_opt": a_opt
        }
        dataset.append(sample)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"\n📁 Gurobi evaluation dataset saved at: {save_path}")

if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_users = 1
    n_antennas_trained = 50
    n_antennas_test = 50
    test_size = 1000
    lr = 1e-4
    draw = 1
    iter = 1
    sigma = 0.3
    K = 20

    parameters_train = get_antenna_parameters(n_antennas_trained, n_users)
    parameters_test = get_antenna_parameters(n_antennas_test, n_users)
    pinch_positions_test = generate_normalized_pinch_positions(parameters=parameters_test)

    test_dataset_path = f"data/val/augmented_dataset_{test_size}samples_{n_antennas_test}ant_SQUARE100_waveguide50_NEWSNR.json"

    snr_acc_gurobi = np.zeros(iter)

    for i in range(iter):
        gurobi_snr_acc, a_opts_gurobi, user_positions_test, B_test_correct, optimal_SNRs_test = test_model_with_gurobi(
            test_size, test_dataset_path, n_antennas_test, n_users,
            pinch_positions_test, parameters_test, dev=dev, sigma=sigma, K=K
        )
        snr_acc_gurobi[i] = gurobi_snr_acc
        print("Gurobi accuracy = ", gurobi_snr_acc)

    avg_active_per_sample, avg_activation_fraction = calculate_antenna_activation(a_opts_gurobi)
    print(f"📡 Average Active Antennas per Sample: {avg_active_per_sample:.2f}")
    print(f"📊 Average Activation Fraction: {avg_activation_fraction:.4f}")
    print("Final  Gurobi SNR Accuracy: ", snr_acc_gurobi.mean())
    print("Final  Gurobi SNR list: ", snr_acc_gurobi)

    # Save results to file in same format as training dataset
    save_path = f"data/train/GurobiNoisyDataset_{test_size}samples_{n_antennas_test}ant_sigma{sigma:.2f}_K{K}.json"
    save_gurobi_eval_dataset(user_positions_test, B_test_correct, optimal_SNRs_test, a_opts_gurobi, parameters_test, save_path)
