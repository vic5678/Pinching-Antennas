from policy_BCE import Policy
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.data import Batch
from validation import validate
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import json


from System_model_setup import beta_calc, calculate_rates, generate_normalized_pinch_positions, get_antenna_parameters, preprocess_data, create_batch_graph,  loss_with_soft_gamma





def save_result_list(result_list, filename):
    with open(filename, 'wb') as f:
        pickle.dump(result_list, f)

def load_result_list(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        return []

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
            


def train(optimal_rates_all,optimal_SNRs_all,a_opt_all, parameters, total_samples, no_antennas, user_positions_all,
          pinch_positions, B_all, policy_net, l_r, no_users, iterations, device,
          batch_size, validation_path, checkpoint_path, save_interval=100, result_save_interval=20, validation_batch_size=100, validate_every=20 ):

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=l_r)
    start_iteration = 0
    best_so_far = 0
    pi = None
    pos_weight_value = 62 / 38  # ≈ 1.63
    pos_weight = torch.tensor(pos_weight_value).to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)
        policy_net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iteration = checkpoint["iteration"] + 1
        best_so_far = checkpoint["best_so_far"]
        validation_accuracies = checkpoint["validation_accuracies"]
        validation_loss_list = checkpoint.get("validation_loss_list", [])
        training_losses = checkpoint.get("training_losses", [])
        training_accuracies = checkpoint.get("training_accuracies", [])
        print(f"🔁 Resuming from iteration {start_iteration} with best accuracy so far: {best_so_far:.4f}")
    else:
        print("🆕 Starting training from scratch.")
        best_so_far = 0
        training_losses = []
        training_accuracies = []
        validation_accuracies = []
        validation_loss_list = []




    num_batches = (total_samples + batch_size - 1) // batch_size

    user_positions_all = user_positions_all[0:total_samples]
    B_all = B_all[0:total_samples]
    optimal_rates_all = np.array(optimal_rates_all)[0:total_samples]
    optimal_SNRs_all = np.array(optimal_SNRs_all)[0:total_samples]

    policy_net.train()

    for itr in range(start_iteration , iterations):
        print(f"🔁 Iteration {itr+1}/{iterations}")

        total_model_ratio = 0
        threshold = 0.9 
        correct_preds = 0 
        iteration_loss = 0
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_samples)

            user_positions_batch = user_positions_all[start_idx:end_idx]
            B_batch = B_all[start_idx:end_idx]
            optimal_rates_batch = optimal_rates_all[start_idx:end_idx]
            optimal_SNRs_batch = optimal_SNRs_all[start_idx:end_idx]
            a_opt_batch = a_opt_all[start_idx:end_idx]

            curr_batch_size = end_idx - start_idx
            user_pos = torch.from_numpy(user_positions_batch).unsqueeze(1).to(torch.float32).to(device)
            B_mag = torch.abs(B_batch).unsqueeze(-1)
            B_phase = torch.angle(B_batch).unsqueeze(-1)
            B_polar = torch.cat([B_mag, B_phase], dim=-1)

            # Graph creation
            batch_graph = create_batch_graph(user_pos = user_pos, pinch_positions = pinch_positions, B = B, B_polar = B_polar, dev = dev)


            pi, imps = policy_net(batch_graph, parameters["N_PINCHES"], curr_batch_size)
            target_actions = torch.tensor(a_opt_batch, dtype=torch.float32).unsqueeze(1)
            #loss = nn.BCELoss()(pi, target_actions)
            # Define loss function
            #loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            # Compute loss using raw logits
            #loss = loss_fn(imps, target_actions)  # both shape [B, N]
            loss = loss_with_soft_gamma(imps, target_actions, B_polar, B_batch, optimal_SNRs_batch, parameters,batch_size, pos_weight_value=1.6, lambda_gamma=10)
                        
            #print("BCE LOSS: ", loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            iteration_loss += loss.item()
            print("Batch ", batch_idx)
            print("Loss: ", loss)

            action_eval = (pi > 0.5).int()
            model_rates, model_SNRs = calculate_rates(B = B_batch, batch_size=curr_batch_size, a_opt=action_eval, parameters=parameters)
            model_SNRs = model_SNRs*10
            model_SNRs = np.array(model_SNRs)
            epsilon = 1e-8
            for i in range(curr_batch_size):
                if optimal_SNRs_batch[i] < epsilon:
                   continue  # Skip bad sample or set accuracy to 0
                #rate_ratio = model_rates[i].item() / optimal_rates_batch[i]
                
                snr_ratio = model_SNRs[i].item() / optimal_SNRs_batch[i]
               
                total_model_ratio += snr_ratio
                if snr_ratio > threshold:
                    correct_preds += 1

        accuracy = correct_preds / total_samples
        average_model_accuracy = total_model_ratio / total_samples
        training_accuracies.append(average_model_accuracy)
        training_losses.append(iteration_loss / num_batches)

        print(f"📊 [Itr {itr+1}] Accuracy = {accuracy:.4f} | Avg Model Accuracy = {average_model_accuracy:.4f} | Loss = {iteration_loss / num_batches:.4f}\n")

        if validation_path and (itr + 1) % validate_every == 0:
            print(f"\n🔍 Running validation at iteration {itr+1}...")
            avg_snr_ratio, avg_val_loss = run_validation(validation_path, policy_net,pinch_positions, parameters,validation_batch_size, no_users, device)
            
            print(f"🧪 [Validation @Itr {itr+1}] Avg SNR Ratio: {avg_snr_ratio:.4f} | Loss: {avg_val_loss:.4f}")

            validation_accuracies.append(avg_snr_ratio)
            validation_loss_list.append(avg_val_loss)

            
            if avg_snr_ratio > best_so_far:
                best_so_far = avg_snr_ratio
                torch.save({
                    'model_state_dict': policy_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iteration': itr,
                    'best_so_far': best_so_far,
                    'validation_accuracies': validation_accuracies,
                    'validation_loss_list': validation_loss_list,
                    'training_accuracies': training_accuracies,
                    'training_losses': training_losses
                }, checkpoint_path)
                print(f"✅ New best model checkpoint saved at iteration {itr+1} with accuracy {avg_snr_ratio:.4f} -> {checkpoint_path}")
            
            policy_net.train()

    return policy, pi, validation_accuracies, training_accuracies


def run_validation(validation_path, policy_net, pinch_positions, parameters, validation_batch_size, no_users, device):
    policy_net.eval()

    with torch.no_grad():
        user_positions_val, B_val_polar, B_val,_, optimal_rates_val, optimal_SNRs_val, a_opts_val = preprocess_data(
            dataset_path=validation_path, total_samples=validation_batch_size, device=device)

        #user_pos_val = torch.from_numpy(user_positions_val).unsqueeze(1).to(torch.float32).to(device)
        
        
        # Create Graph
        batch_graph_val = create_batch_graph(user_pos = torch.from_numpy(user_positions_val).to(torch.float32).to(device), pinch_positions = pinch_positions, B = B_val, B_polar = B_val_polar, dev = dev)

        pi_val, imps_val = policy_net(batch_graph_val, parameters["N_PINCHES"], validation_batch_size)
        action_val = (pi_val > 0.5).int()

        model_rates_val, model_SNRs_val = calculate_rates(
            B_val, batch_size=validation_batch_size, a_opt=action_val, parameters=parameters)

        avg_snr_ratio = np.mean([
            #model_rates_val[i].item() / optimal_rates_val[i]
            #if optimal_rates_val[i] > 1e-8 else 0
            #for i in range(validation_batch_size)
            model_SNRs_val[i].item() / optimal_SNRs_val[i]
            if optimal_SNRs_val[i] > 1e-8 else 0
            for i in range(validation_batch_size)
        ])

        val_target_actions = torch.tensor(a_opts_val, dtype=torch.float32).unsqueeze(1)
        #val_loss = nn.BCELoss()(pi_val, val_target_actions).item()
        val_loss = loss_with_soft_gamma(imps_val, val_target_actions, B_val_polar, B_val, optimal_SNRs_val, parameters, validation_batch_size, pos_weight_value=1.6, lambda_gamma=10)
              


    return avg_snr_ratio, val_loss


# ========= Main ===========
if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(2)
    torch.autograd.set_detect_anomaly(True)

    n_users = 1
    n_antennas = 50
    batch_size = 1000
    total_samples = 5000
    validation_samples = 1000
    test_samples = 1000
    lr = 1e-5
    iterations = 7001
    parameters = get_antenna_parameters(n_antennas, n_users)


    # Load dataset and model
    #train_dataset_path = f"data/train/augmented_dataset_{total_samples}samples_{n_antennas}ant_SQUARE100_waveguide50_NEWSNR.json"
    train_dataset_path = f"data/train/dataset_5000samples50ant_freq28.json"
    #train_dataset_path = f"data/train/GurobiNoisyDataset_1samples_50ant_sigma0.30_K20.json"
    validation_path = f"data/val/dataset_1000samples50ant_freq28.json"
    #checkpoint_path = f"Model_Height3.pth"
    #checkpoint_path = f"dummy.pth"
    checkpoint_path = f"final_freq28.pth"

    pinch_positions = generate_normalized_pinch_positions(parameters)
    user_positions, B_polar, B, _,  optimal_rates,optimal_SNRs, a_opts = preprocess_data(dataset_path=train_dataset_path, total_samples=total_samples, device=dev, use_noisy_B=False)
    user_positions = user_positions.squeeze(1)

    # Policy init
    policy = Policy(in_chnl=1, hid_chnl=128, n_users=n_users, key_size_embd=64,
                    key_size_policy=64, val_size=64, clipping=10, dev=dev).to(dev)

    # Train the model
    model, pi, best_results, result_list = train(optimal_rates,optimal_SNRs, a_opts, parameters, total_samples, n_antennas, user_positions,
                                                 pinch_positions, B, policy, lr, n_users, iterations, dev, batch_size,
                                                 validation_path, checkpoint_path=checkpoint_path, validate_every=1, validation_batch_size=validation_samples)

