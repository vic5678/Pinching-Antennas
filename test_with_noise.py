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
from System_model_setup import plot_antenna_selection, beta_calc, calculate_rates, generate_normalized_pinch_positions, get_antenna_parameters, preprocess_data, create_batch_graph, load_dataset
from GNN_MLP_test import Policy_GNN_MLP

def compute_run_stats(binary_array):
    """Returns number of runs of 1s, max run length, average run length."""
    runs = []
    current_run = 0
    for bit in binary_array:
        if bit == 1:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run)
                current_run = 0
    if current_run > 0:
        runs.append(current_run)
    if not runs:
        return 0, 0, 0.0
    return len(runs), max(runs), sum(runs)/len(runs)

def add_noise(user_pos_test, sigma, K): 
    noise = torch.randn_like(user_pos_test) * sigma
    user_pos_test = user_pos_test + noise
    # Return the noisy user positions
    return user_pos_test

def test_model(policy_test,test_size, test_dataset_path, n_antennas_test, n_users, pinch_positions, parameters,sigma,dev, K):
    # Load test data

    user_positions_test,B_polar_test, B_test_correct,_,optimal_rates_test,optimal_SNRs_test, a_opts_test = preprocess_data(dataset_path=test_dataset_path,total_samples=test_size,device = dev,use_noisy_B=False)
    original_pos_test = torch.from_numpy(user_positions_test).unsqueeze(1).to(torch.float32).to(dev)
    
    # Noisy user position
    #original_noisy_pos_test = add_noise(original_pos_test, sigma)
    
    final_actions = []
    bitwise_accuracies = []
    policy_variances = []

    for i in range(test_size):
        noisy_user_pos = torch.zeros(K,1,3)

        #noisy_user_pos = add_noise(original_pos_test, sigma)
        
        # TAKE K SAMPLES 
        valid_samples = 0
        while valid_samples < K:
            noise = torch.randn_like(original_pos_test[i]) * sigma
            
            candidate = original_pos_test[i] + noise  # shape: [3]
            x, y, z = candidate.squeeze(0).tolist()
            #x, y, z = candidate[0].item(), candidate[1].item(), candidate[2].item()

            if -50 < x < 50 and -50 < y < 50 and 0 < z < 1:
                noisy_user_pos[valid_samples] = candidate
                valid_samples += 1
                #print("x,y,z = ", x, y, z)
            #else:
               #print("Invalid position")

        B_test_noisy = beta_calc(
            parameters["N_PINCHES"],
            pinch_positions,
            noisy_user_pos.squeeze(1).cpu().numpy(),  # make sure shape is (batch_size, 3)
            parameters["LAMBDA"],
            parameters["LAMBDA_G"],
            parameters["PSI_0"],
            batch_size=K
            )

        B_test_noisy = torch.from_numpy(B_test_noisy).to(torch.complex64).to(dev)
        B_mag = torch.abs(B_test_noisy).unsqueeze(-1)
        B_phase = torch.angle(B_test_noisy).unsqueeze(-1)
        B_polar_test = torch.cat([B_mag, B_phase], dim=-1)

        # Graph Creation
        batch_graph_test = create_batch_graph(user_pos = noisy_user_pos, pinch_positions = pinch_positions, B = B_test_noisy, B_polar = B_polar_test,dev = dev)

        # Forward pass
        policy_test.eval()
        with torch.no_grad():
            pi_Ksamples , imp_test = policy_test(batch_graph_test, parameters["N_PINCHES"], K)
            action_Ksamples = (pi_Ksamples > 0.5).int()
        #print(action_Ksamples)

        pi_mean = pi_Ksamples.mean(dim=0,keepdim=True)
        final_action=(pi_mean>0.5).int()
        
        action_Ksamples_float = action_Ksamples.float()
        policy_variance = torch.var(action_Ksamples_float, dim=0).mean().item()
        policy_variances.append(policy_variance)
        
        # Step 2: Compute mean over the samples (dim=0)
        #mean_action = action_Ksamples_float.mean(dim=0, keepdim=True)
        # Step 3: Convert to final binary action using 0.5 threshold
        #final_action = (mean_action > 0.5).int()
        final_actions.append(final_action)

        bitwise_acc = (final_action == torch.from_numpy(a_opts_test[i])).float().mean().item()
        bitwise_accuracies.append(bitwise_acc)
        #print("Bitwise accuracy: ", bitwise_acc)
        #print("final_action:", final_action)
        #print("opt_action:", torch.from_numpy(a_opts_test[i]))

    print("Bitwise accuracy: ", np.mean(bitwise_accuracies))
    print("Policy variance: ", np.mean(policy_variance))
    
    #print("Final actions: ", final_actions)

    # Evaluate accuracy
    final_actions = torch.cat(final_actions, dim=0)
    model_rates_test , model_SNRs_test = calculate_rates(B = B_test_correct, batch_size=test_size,
                                        a_opt=final_actions,
                                        parameters=parameters)

    total_ratio = 0
    total_SNR_ratio=0
    model_total_ones = 0
    model_total_runs = 0
    model_total_max_run = 0
    model_run_lengths = []

    opt_total_ones = 0
    opt_total_runs = 0
    opt_total_max_run = 0
    opt_run_lengths = []
    bad_instances=0
    model_rates_test = np.array(model_rates_test)
    model_SNRs_test = np.array(model_SNRs_test)
    
    for i in range(test_size):
        model_rate = model_rates_test[i].item()
        optimal_rate = optimal_rates_test[i]
        ratio = model_rate / optimal_rate
        total_ratio += ratio

        SNRs_model = model_SNRs_test[i].item()
        optimal_SNR = optimal_SNRs_test[i]
        SNR_ratio = SNRs_model/optimal_SNR
        total_SNR_ratio += SNR_ratio
        #print(SNR_ratio)
        if (SNR_ratio < 0.1):
            bad_instances += 1
        #    print("Actions that resulted in zero rate: ",final_actions[i])
        #    print("User positions that resulted in zero rate: ",user_positions_test[i])
        model_act = final_actions[i].cpu().numpy().flatten().astype(int)
        opt_act = a_opts_test[i].flatten().astype(int)

        # Model stats
        model_ones = np.sum(model_act)
        model_run_count, model_max_run, model_avg_run = compute_run_stats(model_act)

        model_total_ones += model_ones
        model_total_runs += model_run_count
        model_total_max_run += model_max_run
        model_run_lengths.append(model_avg_run)

        # Opt stats
        opt_ones = np.sum(opt_act)
        opt_run_count, opt_max_run, opt_avg_run = compute_run_stats(opt_act)

        opt_total_ones += opt_ones
        opt_total_runs += opt_run_count
        opt_total_max_run += opt_max_run
        opt_run_lengths.append(opt_avg_run)

    # Compute means
    mean_model_ones = model_total_ones / test_size
    mean_model_runs = model_total_runs / test_size
    mean_model_max_run = model_total_max_run / test_size
    mean_model_avg_run = np.mean(model_run_lengths)

    mean_opt_ones = opt_total_ones / test_size
    mean_opt_runs = opt_total_runs / test_size
    mean_opt_max_run = opt_total_max_run / test_size
    mean_opt_avg_run = np.mean(opt_run_lengths)


    # Print summary
    print("\n📊 === AVERAGE STRUCTURAL STATS OVER ALL SAMPLES ===")
    print(f"Model | Ones: {mean_model_ones:.2f}, Runs: {mean_model_runs:.2f}, Max run: {mean_model_max_run:.2f}, Avg run: {mean_model_avg_run:.2f}")
    print(f"Opt   | Ones: {mean_opt_ones:.2f}, Runs: {mean_opt_runs:.2f}, Max run: {mean_opt_max_run:.2f}, Avg run: {mean_opt_avg_run:.2f}")
    print("Bad instances: ", bad_instances)
    
       # print(f"🧪 Test Sample {i+1} | Optimal Action: {a_opts_test[i]} | Action taken: {action_test[i]} | Model Rate: {model_rate:.4f} | Optimal Rate: {optimal_rate:.4f} | Ratio: {ratio:.4f} ")
    test_snr_accuracy= total_SNR_ratio/test_size
    print("SNR Acuracy: ", test_snr_accuracy)
    #print(f"\n🧪 ✅ Average Test SNR Accuracy: {test_snr_accuracy:.4f}")
    test_accuracy = total_ratio / test_size
    #print(f"\n🧪 ✅ Average Test Accuracy: {test_accuracy:.4f}")

    # Count total active antennas across all test samples
    total_active_model = final_actions.sum().item()
    total_active_opt = np.sum(a_opts_test)

    avg_active_model = total_active_model / (test_size * n_antennas_test)
    print("Average model antenna activarion: ", avg_active_model)
    avg_active_opt = total_active_opt / (test_size * n_antennas_test)

    #print(f"\n📡 Total Active Antennas (Model): {total_active_model}/{test_size * n_antennas_test} "
     #     f"({total_active_model / (test_size * n_antennas_test):.4f})")
    #print(f"📡 Total Active Antennas (Opt)  : {total_active_opt}/{test_size * n_antennas_test} "
     #     f"({total_active_opt / (test_size * n_antennas_test):.4f})")
    #print(f"📈 Avg. Active Antennas per Sample (Model): {avg_active_model:.2f}")
    #print(f"📈 Avg. Active Antennas per Sample (Opt)  : {avg_active_opt:.2f}")


    
    return test_snr_accuracy,test_accuracy, final_actions, user_positions_test, avg_active_model, avg_active_opt


def test_model_GNN_MLP(policy_test,test_size, test_dataset_path, n_antennas_test, n_users, pinch_positions, parameters,sigma,dev, K):
    # Load test data

    user_positions_test,B_polar_test, B_test_correct,_,optimal_rates_test,optimal_SNRs_test, a_opts_test = preprocess_data(dataset_path=test_dataset_path,total_samples=test_size,device = dev,use_noisy_B=False)
    original_pos_test = torch.from_numpy(user_positions_test).unsqueeze(1).to(torch.float32).to(dev)
    
    # Noisy user position
    #original_noisy_pos_test = add_noise(original_pos_test, sigma)
    
    final_actions = []
    bitwise_accuracies = []
    policy_variances = []

    for i in range(test_size):
        noisy_user_pos = torch.zeros(K,1,3)

        #noisy_user_pos = add_noise(original_pos_test, sigma)
        
        # TAKE K SAMPLES 
        valid_samples = 0
        while valid_samples < K:
            noise = torch.randn_like(original_pos_test[i]) * sigma
            
            candidate = original_pos_test[i] + noise  # shape: [3]
            x, y, z = candidate.squeeze(0).tolist()
            #x, y, z = candidate[0].item(), candidate[1].item(), candidate[2].item()

            if -50 < x < 50 and -50 < y < 50 and 0 < z < 1:
                noisy_user_pos[valid_samples] = candidate
                valid_samples += 1
                #print("x,y,z = ", x, y, z)
            #else:
               #print("Invalid position")

        B_test_noisy = beta_calc(
            parameters["N_PINCHES"],
            pinch_positions,
            noisy_user_pos.squeeze(1).cpu().numpy(),  # make sure shape is (batch_size, 3)
            parameters["LAMBDA"],
            parameters["LAMBDA_G"],
            parameters["PSI_0"],
            batch_size=K
            )

        B_test_noisy = torch.from_numpy(B_test_noisy).to(torch.complex64).to(dev)
        B_mag = torch.abs(B_test_noisy).unsqueeze(-1)
        B_phase = torch.angle(B_test_noisy).unsqueeze(-1)
        B_polar_test = torch.cat([B_mag, B_phase], dim=-1)

        # Graph Creation
        batch_graph_test = create_batch_graph(user_pos = noisy_user_pos, pinch_positions = pinch_positions, B = B_test_noisy, B_polar = B_polar_test,dev = dev)

        # Forward pass
        policy_test.eval()
        with torch.no_grad():
            pi_Ksamples , imp_test = policy_test(batch_graph_test, parameters["N_PINCHES"], K)
            action_Ksamples = (pi_Ksamples > 0.5).int()
        #print(action_Ksamples)

        pi_mean = pi_Ksamples.mean(dim=0,keepdim=True)
        final_action=(pi_mean>0.5).int()
        
        action_Ksamples_float = action_Ksamples.float()
        policy_variance = torch.var(action_Ksamples_float, dim=0).mean().item()
        policy_variances.append(policy_variance)
        
        # Step 2: Compute mean over the samples (dim=0)
        #mean_action = action_Ksamples_float.mean(dim=0, keepdim=True)
        # Step 3: Convert to final binary action using 0.5 threshold
        #final_action = (mean_action > 0.5).int()
        final_actions.append(final_action)

        bitwise_acc = (final_action == torch.from_numpy(a_opts_test[i])).float().mean().item()
        bitwise_accuracies.append(bitwise_acc)
        #print("Bitwise accuracy: ", bitwise_acc)
        #print("final_action:", final_action)
        #print("opt_action:", torch.from_numpy(a_opts_test[i]))

    print("Bitwise accuracy: ", np.mean(bitwise_accuracies))
    print("Policy variance: ", np.mean(policy_variance))
    
    #print("Final actions: ", final_actions)

    # Evaluate accuracy
    final_actions = torch.cat(final_actions, dim=0)
    model_rates_test , model_SNRs_test = calculate_rates(B = B_test_correct, batch_size=test_size,
                                        a_opt=final_actions,
                                        parameters=parameters)

    total_ratio = 0
    total_SNR_ratio=0
    model_total_ones = 0
    model_total_runs = 0
    model_total_max_run = 0
    model_run_lengths = []

    opt_total_ones = 0
    opt_total_runs = 0
    opt_total_max_run = 0
    opt_run_lengths = []
    bad_instances=0
    model_rates_test = np.array(model_rates_test)
    model_SNRs_test = np.array(model_SNRs_test)
    
    for i in range(test_size):
        model_rate = model_rates_test[i].item()
        optimal_rate = optimal_rates_test[i]
        ratio = model_rate / optimal_rate
        total_ratio += ratio

        SNRs_model = model_SNRs_test[i].item()
        optimal_SNR = optimal_SNRs_test[i]
        SNR_ratio = SNRs_model/optimal_SNR
        total_SNR_ratio += SNR_ratio
        #print(SNR_ratio)
        if (SNR_ratio < 0.1):
            bad_instances += 1
        #    print("Actions that resulted in zero rate: ",final_actions[i])
        #    print("User positions that resulted in zero rate: ",user_positions_test[i])
        model_act = final_actions[i].cpu().numpy().flatten().astype(int)
        opt_act = a_opts_test[i].flatten().astype(int)

        # Model stats
        model_ones = np.sum(model_act)
        model_run_count, model_max_run, model_avg_run = compute_run_stats(model_act)

        model_total_ones += model_ones
        model_total_runs += model_run_count
        model_total_max_run += model_max_run
        model_run_lengths.append(model_avg_run)

        # Opt stats
        opt_ones = np.sum(opt_act)
        opt_run_count, opt_max_run, opt_avg_run = compute_run_stats(opt_act)

        opt_total_ones += opt_ones
        opt_total_runs += opt_run_count
        opt_total_max_run += opt_max_run
        opt_run_lengths.append(opt_avg_run)

    # Compute means
    mean_model_ones = model_total_ones / test_size
    mean_model_runs = model_total_runs / test_size
    mean_model_max_run = model_total_max_run / test_size
    mean_model_avg_run = np.mean(model_run_lengths)

    mean_opt_ones = opt_total_ones / test_size
    mean_opt_runs = opt_total_runs / test_size
    mean_opt_max_run = opt_total_max_run / test_size
    mean_opt_avg_run = np.mean(opt_run_lengths)


    # Print summary
    print("\n📊 === AVERAGE STRUCTURAL STATS OVER ALL SAMPLES ===")
    print(f"Model | Ones: {mean_model_ones:.2f}, Runs: {mean_model_runs:.2f}, Max run: {mean_model_max_run:.2f}, Avg run: {mean_model_avg_run:.2f}")
    print(f"Opt   | Ones: {mean_opt_ones:.2f}, Runs: {mean_opt_runs:.2f}, Max run: {mean_opt_max_run:.2f}, Avg run: {mean_opt_avg_run:.2f}")
    print("Bad instances: ", bad_instances)
    
       # print(f"🧪 Test Sample {i+1} | Optimal Action: {a_opts_test[i]} | Action taken: {action_test[i]} | Model Rate: {model_rate:.4f} | Optimal Rate: {optimal_rate:.4f} | Ratio: {ratio:.4f} ")
    test_snr_accuracy= total_SNR_ratio/test_size
    print("SNR Acuracy: ", test_snr_accuracy)
    #print(f"\n🧪 ✅ Average Test SNR Accuracy: {test_snr_accuracy:.4f}")
    test_accuracy = total_ratio / test_size
    #print(f"\n🧪 ✅ Average Test Accuracy: {test_accuracy:.4f}")

    # Count total active antennas across all test samples
    total_active_model = final_actions.sum().item()
    total_active_opt = np.sum(a_opts_test)

    avg_active_model = total_active_model / (test_size * n_antennas_test)
    print("Average model antenna activarion: ", avg_active_model)
    avg_active_opt = total_active_opt / (test_size * n_antennas_test)

    #print(f"\n📡 Total Active Antennas (Model): {total_active_model}/{test_size * n_antennas_test} "
     #     f"({total_active_model / (test_size * n_antennas_test):.4f})")
    #print(f"📡 Total Active Antennas (Opt)  : {total_active_opt}/{test_size * n_antennas_test} "
     #     f"({total_active_opt / (test_size * n_antennas_test):.4f})")
    #print(f"📈 Avg. Active Antennas per Sample (Model): {avg_active_model:.2f}")
    #print(f"📈 Avg. Active Antennas per Sample (Opt)  : {avg_active_opt:.2f}")


    
    return test_snr_accuracy,test_accuracy, final_actions, user_positions_test, avg_active_model, avg_active_opt



if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_users = 1
    n_antennas_trained = 50
    n_antennas_test = 50
    test_size = 100
    lr = 1e-3
    draw = 1 
    iter = 1
    K = 100

    torch.manual_seed(42)
    np.random.seed(42)

    # Parameters based on training config
    parameters_train = get_antenna_parameters(n_antennas_trained, n_users)
    
    # Parameters based on test config
    parameters_test = get_antenna_parameters(n_antennas_test, n_users)
    pinch_positions_test = generate_normalized_pinch_positions(parameters=parameters_test)

    # Load dataset and model
    test_dataset_path = f"data/test/augmented_dataset_{test_size}samples_{n_antennas_test}ant_SQUARE100_waveguide50_NEWSNR.json"
    
    policy = Policy(in_chnl=1, hid_chnl=128, n_users=n_users, key_size_embd=64,
               key_size_policy=64, val_size=64, clipping=10, dev=dev).to(dev)

    #model_path = f"checkpoint_GNN+DispN_1Layer_50ant_1user_lr_0.001_BCE_5000_samples_hid64+128_SQUARE100_waveguide50_edgeBased.pth"
    model_path = f"Model_Hybrid_loss_lambda2.pth"
  
    #policy = Policy_GNN_MLP( in_chnl=1,  hid_chnl=64, mlp_hidden_dim=128, dev=dev)
    #model_path = f"checkpoint_GNN+MLP_{n_antennas_trained}ant_{n_users}user_lr_{lr}_BCE_5000_samples_SQUARE100_waveguide50_hid64+128.pth"

    checkpoint = torch.load(model_path, map_location=dev, weights_only=False)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    #sigmas = np.arange(0.01, 0.51, 0.05)
    sigmas = np.arange(0.0, 1.0, 0.05)
    snr_accuracies = []
    model_antenna_activation = []
    optimal_antenna_activation = []
    accuracies_for_sigma = []



    for sigma in sigmas:
        accuracies_for_sigma = []
        avg_active_model_list = []
        avg_active_opt_list = []
        
        print(f"\n🔎 Testing for sigma={sigma:.2f}")

        for repeat in range(iter):  # Repeat to average out randomness
            print("ITERATION: ", repeat)
            snr_acc,test_accuracy, action_test, user_positions_test, avg_active_model, avg_active_opt =test_model(policy, test_size, test_dataset_path, n_antennas_test, n_users,
                                           pinch_positions_test, parameters_test,
                                            sigma=sigma, dev=dev, K = K)
            accuracies_for_sigma.append(snr_acc)
            avg_active_model_list.append(avg_active_model)
            print("avg_active_model_list: ",avg_active_model_list)
            print("Accuracy: ", accuracies_for_sigma)
            avg_active_opt_list.append(avg_active_opt)
        
        avg_snr_acc = np.mean(accuracies_for_sigma)
        total_avg_active_model = np.mean(avg_active_model_list)
        total_avg_active_opt = np.mean(avg_active_opt_list)
       
        snr_accuracies.append(avg_snr_acc)
        model_antenna_activation.append(total_avg_active_model)
        optimal_antenna_activation.append(total_avg_active_opt)

        print(f"✅ Average SNR Accuracy @ sigma={sigma:.2f}: {avg_snr_acc:.4f}")
        print(f"✅ Average Optimal Active antennas @ sigma={sigma:.2f}: {total_avg_active_opt:.4f}")
        print(f"✅ Average Model Active antennas @ sigma={sigma:.2f}: {total_avg_active_model:.4f}")
    
    print("SNR accuracies: ", snr_accuracies)

    plt.figure(figsize=(8, 5))
    plt.plot(sigmas, model_antenna_activation, marker='o', label='Model Active Antennas')
    plt.plot(sigmas, optimal_antenna_activation, marker='x', linestyle='--', label='Optimal Active Antennas')
    plt.xlabel('Sigma (User Position Noise)')
    plt.ylabel('Avg. Fraction of Active Antennas')
    plt.title('Antenna Activation vs. Sigma')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()


    


    