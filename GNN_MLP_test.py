import torch
from torch.distributions import Categorical
from policy_GNN_MLP import Policy_GNN_MLP
from torch_geometric.data import Batch , Data
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import torch.nn as nn
from System_model_setup import  calculate_rates, generate_normalized_pinch_positions, get_antenna_parameters, plot_antenna_selection, load_dataset, preprocess_data, beta_calc, create_batch_graph


def test_model(policy_test,test_size, test_dataset_path, n_antennas_test, n_users, pinch_positions, parameters,dev):
    
    # Load test data
    user_positions_test,B_polar_test, B_test,_,optimal_rates_test,optimal_SNRs_test, a_opts_test = preprocess_data(dataset_path=test_dataset_path,total_samples=test_size,device = dev)
    user_pos_test = torch.from_numpy(user_positions_test).unsqueeze(1).to(torch.float32).to(dev)

    # Graph Creation
    batch_graph_test = create_batch_graph(user_pos = user_pos_test, pinch_positions = pinch_positions, B = B_test, B_polar = B_polar_test,dev = dev)

    # Forward pass
    policy_test.eval()
    with torch.no_grad():

        start_time = time.time()
        pi_test, imp = policy_test(batch_graph_test, parameters["N_PINCHES"], test_size)
        #print("PI TEST: ", pi_test[0])
        # ✅ Prepare ground truth tensor for BCE
        target_actions_test = torch.tensor(a_opts_test, dtype=torch.float32).to(pi_test.device).unsqueeze(1)
        #print("target_actions_test: ", target_actions_test[0])
        
        # ✅ Compute BCE loss
        bce_loss = nn.BCELoss()(pi_test, target_actions_test)
        
        action_test = (pi_test > 0.5).int()
        end_time = time.time()
        forward_pass_time = end_time - start_time
        
        #print("pi_test.shape[-1]", pi_test.shape[-1])
    
    # Evaluate accuracy
    model_rates_test, SNRs_test = calculate_rates(B = B_test, batch_size=test_size,
                                       a_opt=action_test,
                                       parameters=parameters)
    
    total_ratio = 0
    total_SNR_ratio = 0
    same_action = 0
    correct_bits = 0
    total_bits = 0
    for i in range(test_size):

        model_rate = model_rates_test[i].item()
        optimal_rate = float(optimal_rates_test[i])
        ratio = model_rate / optimal_rate
        total_ratio += ratio

        SNRs_rate = SNRs_test[i].item()
        optimal_SNR = optimal_SNRs_test[i]
        SNR_ratio = SNRs_rate/optimal_SNR
        total_SNR_ratio+= SNR_ratio

        pred_action = action_test[i].cpu().numpy().flatten()
        true_action = np.array(a_opts_test[i]).flatten()

        correct_bits += np.sum(pred_action == true_action)
        total_bits += len(true_action)
        model_action = action_test[i].cpu().numpy().reshape(-1)
        true_action = np.array(a_opts_test[i]).reshape(-1)
        if np.array_equal(model_action, true_action):
            #print("Same action")
            same_action+=1
        #else:
             #print("Different Action")
        

        #print(f"🧪 Test Sample {i+1} | Optimal Action: {a_opts_test[i]} | Action taken: {action_test[i]} | Model Rate: {float(model_rate):.4f} | Optimal Rate: {float(optimal_rate):.4f} | Ratio: {float(ratio):.4f}")
    print(f"🧪 Test BCE Loss: {bce_loss.item():.6f}")
    print("Model Rates mean: ", model_rates_test.mean())
    print("Optimal Rates mean: ", np.array(optimal_rates_test).mean())
    mse_rate_loss = np.mean((optimal_rate - model_rate) ** 2)
    print("Total MSE: ", mse_rate_loss)
    test_rate_accuracy = total_ratio / test_size
    test_bce_accuracy = same_action / test_size
    bitwise_accuracy = correct_bits / total_bits
    test_snr_accuracy= total_SNR_ratio/test_size

    # Total antennas active
    total_active_model = action_test.sum().item()
    avg_active_model = total_active_model / (test_size * n_antennas_test)

    print(f"\n🧪 ✅ Average Test SNR Accuracy: {test_snr_accuracy:.4f}")

    print(f"\n🧪 ✅ Average Test Rate Accuracy: {test_rate_accuracy:.4f}")
    print(f"\n🧪 ✅ Average Test BCE Accuracy (How many actions were 100% optimal): {test_bce_accuracy:.4f}")
    print(f"\n🔢 BCE Bitwise Accuracy over test set: {bitwise_accuracy:.4f}")
    print(f"⏱️ Forward Pass Time: {forward_pass_time:.6f} seconds")

    print(f"{same_action} SAME ACTIONS !")
    return test_rate_accuracy, action_test, user_positions_test, avg_active_model

if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_users = 1
    n_antennas_trained = 50
    n_antennas_test = 500
    test_size = 50
    lr= 1e-3
    draw = 1 

    # Parameters based on training config
    parameters_train = get_antenna_parameters(n_antennas_trained, n_users)

    # Parameters based on test config
    parameters_test = get_antenna_parameters(n_antennas_test, n_users)
    pinch_positions_test = generate_normalized_pinch_positions(parameters=parameters_test)

    # Load dataset and model
    #test_dataset_path = f"data/test/augmented_dataset_{test_size}samples_{n_antennas_test}ant_SQUARE100_waveguide50_NEWSNR.json"
    test_dataset_path = f"data/test/augmented_dataset_{test_size}samples_{n_antennas_test}ant_SQUARE100_waveguide50_NEWSNR.json"
    #policy = Policy_GNN_MLP(in_chnl=3, hid_chnl=64, mlp_hidden_dim=128, dev=dev )    
    policy = Policy_GNN_MLP( in_chnl=1,  hid_chnl=64, mlp_hidden_dim=128, dev=dev)
    model_path = f"checkpoint_GNN+MLP_{n_antennas_trained}ant_{n_users}user_lr_{lr}_BCE_5000_samples_SQUARE100_waveguide50_hid64+128.pth"
    #model_path = f"checkpoint_GNN+MLP_ManyNoises.pth"
    checkpoint = torch.load(model_path, map_location=dev, weights_only=False)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    # Test the model
    acc , a_opts_test, user_positions_test = test_model(policy,test_size,  test_dataset_path, n_antennas_test, n_users, pinch_positions_test, parameters_test,dev=dev)

    # Plot the selected antennas for sample 0
    sample_idx = 0
    user_pos_sample = user_positions_test[sample_idx]
    selected_antennas = a_opts_test[sample_idx].cpu().numpy().flatten()

    plot_antenna_selection(user_pos_sample, pinch_positions_test, selected_antennas,
                           title="Model Antenna Selection")
    plt.show()
    