import torch
from torch.distributions import Categorical
from policy_BCE import Policy
from torch_geometric.data import Batch , Data
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import torch.nn as nn
from System_model_setup import calculate_rates_torch,plot_antenna_selection, generate_normalized_pinch_positions, get_antenna_parameters, beta_calc, create_batch_graph, load_dataset, preprocess_data
from GNN_MLP_test import test_model as GNN_MLP_Test, Policy_GNN_MLP


def test_model(policy_test,test_size, test_dataset_path, n_antennas_test, n_users, pinch_positions, parameters,dev):
    # Load test data


    user_positions_test,B_polar_test, B_test,_,optimal_rates_test,optimal_SNRs_test, a_opts_test = preprocess_data(dataset_path=test_dataset_path,total_samples=test_size,device = dev, use_noisy_B=False)
    #user_pos_test = torch.from_numpy(user_positions_test).unsqueeze(1).to(torch.float32).to(dev)
    user_pos_test = torch.from_numpy(user_positions_test).to(torch.float32).to(dev)

    # Graph Creation
    batch_graph_test = create_batch_graph(user_pos = user_pos_test, pinch_positions = pinch_positions, B = B_test, B_polar = B_polar_test,dev = dev)
    print("batch_graph_test: ", batch_graph_test)
    print("PINSH POSITIONS = " , pinch_positions)
    # Forward pass
    policy_test.eval()
    with torch.no_grad():
        print("N_ANNTENNAS = ", parameters["N_PINCHES"])
        start_time = time.time()
        pi_test, imp = policy_test(batch_graph_test, parameters["N_PINCHES"], test_size)
        #print("PI TEST: ", pi_test[0])
        
        # ✅ Prepare ground truth tensor for BCE
        target_actions_test = torch.tensor(a_opts_test, dtype=torch.float32).unsqueeze(1)
        #print("target_actions_test: ", target_actions_test[0])
        
        # ✅ Compute BCE loss
        bce_loss = nn.BCELoss()(pi_test, target_actions_test)


        
        action_test = (pi_test > 0.5).int()
        end_time = time.time()
        forward_pass_time = end_time - start_time
        
        #print("pi_test.shape[-1]", pi_test.shape[-1])
    
    # Evaluate accuracy
    model_rates_test, SNRs_test = calculate_rates_torch(B = B_test, batch_size=test_size,
                                       a_opt=action_test,
                                       parameters=parameters)


    
    total_ratio = 0
    same_action = 0
    correct_bits = 0
    total_bits = 0
    total_SNR_ratio =0
    bad_instances = 0
    for i in range(test_size):

        model_rate = model_rates_test[i].item()
        optimal_rate = float(optimal_rates_test[i])
        ratio = model_rate / optimal_rate
        total_ratio += ratio
        
        SNRs_rate = SNRs_test[i].item()
        optimal_SNR = optimal_SNRs_test[i]
        SNR_ratio = SNRs_rate/optimal_SNR
        if SNR_ratio < 0.2:
            #print(f"BAD INSTANCE at index {i} with ratio  {SNR_ratio} , action: {action_test[i]}. Optimal actions: {a_opts_test[i]}")
            bad_instances += 1
        total_SNR_ratio+= SNR_ratio

        pred_action = action_test[i].cpu().numpy().flatten()
        true_action = np.array(a_opts_test[i]).flatten()

        correct_bits += np.sum(pred_action == true_action)
        total_bits += len(true_action)
        model_action = action_test[i].cpu().numpy().reshape(-1)
        print("MODEL: ", model_action)
        print("MODEL SNR: ", SNRs_rate)
        print("OPTIM: ", true_action)
        print("OPTIM SNR: ",optimal_SNR)
        true_action = np.array(a_opts_test[i]).reshape(-1)
        if np.array_equal(model_action, true_action):
            print("Same action")
            same_action+=1
        else:
            print("Different Action")
        

        #print(f"🧪 Test Sample {i+1} | Optimal Action: {a_opts_test[i]} | Action taken: {action_test[i]} | Model Rate: {float(model_rate):.4f} | Optimal Rate: {float(optimal_rate):.4f} | Ratio: {float(ratio):.4f}")
    #print(f"🧪 Test BCE Loss: {bce_loss.item():.6f}")
    mse_rate_loss = np.mean((optimal_rate - model_rate) ** 2)
    #print("Total MSE: ", mse_rate_loss)
    test_rate_accuracy = total_ratio / test_size
    test_bce_accuracy = same_action / test_size
    bitwise_accuracy = correct_bits / total_bits
    test_snr_accuracy = total_SNR_ratio/test_size
    print(f"\n🧪 ✅ Average Test SNR Accuracy: {test_snr_accuracy:.4f}")
    #print(f"\n🧪 ✅ Average Test Rate Accuracy: {test_rate_accuracy:.4f}")
    #print(f"\n🧪 ✅ Average Test BCE Accuracy (How many actions were 100% optimal): {test_bce_accuracy:.4f}")
    #print(f"\n🔢 BCE Bitwise Accuracy over test set: {bitwise_accuracy:.4f}")
    #print(f"⏱️ Forward Pass Time: {forward_pass_time:.6f} seconds")
    print(f"BAD instances: {bad_instances}")
    # Count total active antennas across all test samples
    total_active_model = action_test.sum().item()
    total_active_opt = np.sum(a_opts_test)

    avg_active_model = total_active_model / (test_size * n_antennas_test)
    avg_active_opt = total_active_opt / (test_size * n_antennas_test)


    #print(f"📈 Avg. Active Antennas (Model): {avg_active_model:.2f}")
    #print(f"📈 Avg. Active Antennas (Opt)  : {avg_active_opt:.2f}")



    #print(f"{same_action} SAME ACTIONS !")

     #### HERE REMOVE SNRS_test ####
    return test_snr_accuracy, test_rate_accuracy, action_test, user_positions_test, avg_active_model, avg_active_opt 

if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_users = 1
    n_antennas_trained = 50
    #n_antennas_test = np.array([500])
    test_size = 1000
    lr = 1e-3
    draw = 1 
    #n_antennas_test = np.array([10,20,50,100,200,500,1000])
    #n_antennas_test = np.array([20,50,100,200,300,500,1000])
    n_antennas_test = np.array([50])
    
    

    # GNN+DispN
    policy = Policy(in_chnl=1, hid_chnl=128, n_users=n_users, key_size_embd=64, key_size_policy=64, val_size=64, clipping=10, dev=dev).to(dev)
    #model_path = f"checkpoint_GNN+DispN_1Layer_{n_antennas_trained}ant_{n_users}user_lr_{lr}_BCE_5000_samples_hid64+128_SQUARE100_waveguide50_edgeBased.pth"
    # THIS IS THE BEST MODEL
    model_path = f"Model_Hybrid_loss_lambda2.pth"

    # GNN_MLP
    policy_GNN_mlp = Policy_GNN_MLP( in_chnl=1,  hid_chnl=64, mlp_hidden_dim=128, dev=dev)
    model_path_GNN_mlp = f"checkpoint_GNN+MLP_{n_antennas_trained}ant_{n_users}user_lr_{lr}_BCE_5000_samples_SQUARE100_waveguide50_hid64+128.pth"

    #model_path = f"Model_Hybrid_loss_lambda2.pth"
    #model_path = f"dummy.pth"
    #model_path =f"Model_Weighted_loss.pth"
    #model_path = f"checkpoint_GNN+DispN_{n_antennas_trained}ant_1user_lr_{lr}_BCE_5000_samples_SQUARE100_STOCHASTIC_hid128+64.pth"
    #model_path = f"checkpoint_GNN+DispN_ManyNoises.pth"
    #model_path = f"checkpoint_GNN+DispN_noise0.5_stableDataset.pth"
    #model_path = f"final_model_correct.pth"
    checkpoint = torch.load(model_path, map_location=dev, weights_only=False)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    checkpoint_gnn_mlp = torch.load(model_path_GNN_mlp, map_location=dev, weights_only=False)

    policy_GNN_mlp.load_state_dict(checkpoint_gnn_mlp["model_state_dict"])
    policy_GNN_mlp.eval()

    avg_activation_model = []
    avg_activation_opt = []
    avg_activation_gnn_mlp = []

    for size in n_antennas_test:


        # Parameters based on training config
        parameters_train = get_antenna_parameters(n_antennas_trained, n_users)

        # Parameters based on test config
        parameters_test = get_antenna_parameters(size, n_users)
        
        pinch_positions_test = generate_normalized_pinch_positions(parameters=parameters_test)
        print("PINCH POSITIONS = ", pinch_positions_test)

        # Load dataset and model
        #test_dataset_path = f"data/test/augmented_dataset_{test_size}samples_{size}ant_SQUARE100_waveguide50_NEWSNR.json"
        
        #test_dataset_path = f"data/test/dataset_1samples45ant_OOD.json"

        #test_dataset_path = f"data/test/dataset_1000samples50ant_BLOCKAGE_ratio90.json"

        test_dataset_path = f"data/test/dataset_1000samples50ant_BLOCKAGE_softmax_95ratio.json"
        
        #test_dataset_path = f"data/test/augmented_dataset_100samples_50ant_SQUARE100_waveguide50_NEWSNR.json"
        

        # Test the model
        snr_acc, acc , a_opts_test, user_positions_test, avg_active_model, avg_active_opt = test_model(policy,test_size,  test_dataset_path, size, n_users, pinch_positions_test, parameters_test,dev=dev)

        # Test GNN MLP
        #acc_gnn_mlp , a_opts_test_gnn_mlp, user_positions_test,avg_active_gnn_mlp = GNN_MLP_Test(policy_GNN_mlp,test_size,  test_dataset_path, size, n_users, pinch_positions_test, parameters_test,dev=dev)

        avg_activation_model.append(avg_active_model)
        #avg_activation_gnn_mlp.append(avg_active_gnn_mlp )
        avg_activation_opt.append(avg_active_opt)
    print("avg_activation_gnn_mlp: ", avg_activation_gnn_mlp)
    print("avg_activation model: ", avg_activation_model)
    print("snr acc of model = ", snr_acc)

    # Plot
    """
    plt.figure(figsize=(8, 5))
    plt.plot(n_antennas_test, avg_activation_model, color='b', label='Model')
    plt.plot(n_antennas_test, avg_activation_opt, color='g', label='Optimal')
    plt.plot(n_antennas_test, avg_activation_gnn_mlp, color='r', label='GNN-MLP')
    plt.xlabel('Number of Antennas')
    plt.ylabel('Antenna Activation')
    plt.title('Antenna Activation for Different Antenna Counts')
    plt.grid(True)
    plt.legend()  # ✅ This enables the legend
    plt.show()
    

        
    # Plot the selected antennas for sample 0
    sample_idx = 0
    user_pos_sample = user_positions_test[sample_idx]
    selected_antennas = a_opts_test[sample_idx].cpu().numpy().flatten()

    #plot_antenna_selection(user_pos_sample, pinch_positions_test, selected_antennas,
    #                       title="Model Antenna Selection")
    

    #plt.show()
    data_to_save = np.column_stack((n_antennas_test, avg_activation_gnn_mlp))
    data_to_save1 = np.column_stack((n_antennas_test, avg_activation_opt))
    data_to_save2 = np.column_stack((n_antennas_test, avg_activation_model))

# Save as .dat file
    np.savetxt('n_ant_test_vs_avg_activation_model_gnn_mlp.dat', data_to_save, fmt='%.5f', delimiter='\t')
    np.savetxt('n_ant_test_vs_avg_activation_opt.dat', data_to_save1, fmt='%.5f', delimiter='\t')
    np.savetxt('n_ant_test_vs_avg_activation_model.dat', data_to_save2, fmt='%.5f', delimiter='\t')


    print("✅ Data saved") 
    """