from policy_GNN_MLP import Policy_GNN_MLP
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


from System_model_setup import beta_calc, calculate_rates, generate_normalized_pinch_positions, get_antenna_parameters, preprocess_data, create_batch_graph


def make_dummy_x(n_users, n_antennas, dev):
    x = torch.ones((n_users + n_antennas, 3), dtype=torch.float32).to(dev)
    x[:, 0] = 0.0  # all antennas
    x[:n_users, 0] = 1.0  # user nodes
    return x


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

def train(optimal_rates_all,a_opt_all, parameters, total_samples, no_antennas, user_positions_all,
          pinch_positions, B_all, policy_net, l_r, no_users, iterations, device,
          batch_size, validation_path, checkpoint_path, save_interval=100, result_save_interval=20, validation_batch_size=100, validate_every=1 ):
    #print("Training Beta's: ", B_all[0])
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=l_r)
    start_iteration = 0
    best_so_far = 0
    pi = None

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)
        policy_net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iteration = checkpoint["iteration"] + 1
        best_so_far = checkpoint["best_so_far"]
        validation_results = checkpoint["validation_results"]
        result_list = checkpoint["result_list"]
        print(f"🔁 Resuming from iteration {start_iteration} with best accuracy so far: {best_so_far:.4f}")
    else:
        print("🆕 Starting training from scratch.")
        best_so_far = 0
        result_list = []
        validation_results = []
    

    num_batches = (total_samples + batch_size - 1) // batch_size

    user_positions_all = user_positions_all[0:total_samples]
    B_all = B_all[0:total_samples]
    optimal_rates_all = np.array(optimal_rates_all)[0:total_samples]
    x_dummy = make_dummy_x(n_users, n_antennas, dev)

    policy_net.train()

    for itr in range(start_iteration , iterations):
        print(f"🔁 Iteration {itr+1}/{iterations}")

        total_model_ratio = 0
        threshold = 0.9 
        correct_preds = 0 

        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_samples)

            user_positions_batch = user_positions_all[start_idx:end_idx]
            B_batch = B_all[start_idx:end_idx]
            optimal_rates_batch = optimal_rates_all[start_idx:end_idx]
            a_opt_batch = a_opt_all[start_idx:end_idx]

            curr_batch_size = end_idx - start_idx
            user_pos = torch.from_numpy(user_positions_batch).unsqueeze(1).to(torch.float32).to(device)
            B_mag = torch.abs(B_batch).unsqueeze(-1)
            B_phase = torch.angle(B_batch).unsqueeze(-1)
            B_polar = torch.cat([B_mag, B_phase], dim=-1)
            batch_graph = create_batch_graph(user_pos = user_pos, pinch_positions = pinch_positions, B = B, B_polar = B_polar, dev = dev)

            pi, imps = policy_net(batch_graph, parameters["N_PINCHES"], curr_batch_size)
            target_actions = torch.tensor(a_opt_batch, dtype=torch.float32).to(pi.device).unsqueeze(1)
            loss = nn.BCELoss()(pi, target_actions)
            print("BCE LOSS: ", loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            result_list.append(loss)
            print("Batch ", batch_idx)
            print("Loss: ", loss)

            action_eval = (pi > 0.5).int()
            model_rates, SNRs = calculate_rates(B = B_batch, batch_size=curr_batch_size, a_opt=action_eval, parameters=parameters)
            
            epsilon = 1e-8
            for i in range(curr_batch_size):
                if optimal_rates_batch[i] < epsilon:
                   continue  # Skip bad sample or set accuracy to 0
                rate_ratio = model_rates[i].item() / optimal_rates_batch[i]
                total_model_ratio += rate_ratio
                if rate_ratio > threshold:
                    correct_preds += 1

        accuracy = correct_preds / total_samples
        average_model_accuracy = total_model_ratio / total_samples
        print(f"📊 [Itr {itr+1}] Accuracy = {accuracy:.4f} | Avg Model Accuracy = {average_model_accuracy:.4f}\n")

        if validation_path and (itr + 1) % validate_every == 0:
            policy_net.eval()
            print(f"\n🔍 Running validation at iteration {itr+1}...")
            avg_val_ratio = run_validation(validation_path,policy_net,
                                           pinch_positions, parameters, validation_batch_size, no_users, device)
            print(f"🧪 [Validation @Itr {itr+1}] Avg Accuracy: {avg_val_ratio:.4f}")
            validation_results.append(avg_val_ratio)
            """
            if avg_val_ratio > best_so_far:
                best_so_far = avg_val_ratio
                torch.save({
                    'model_state_dict': policy_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iteration': itr,
                    'best_so_far': best_so_far,
                    'validation_results': validation_results,
                    'result_list': result_list
                }, checkpoint_path)
                print(f"✅ New best model checkpoint saved at iteration {itr+1} with accuracy {avg_val_ratio:.4f} -> {checkpoint_path}")
                    """
            policy_net.train()

    return policy, pi, validation_results, result_list
def run_validation(validation_path, policy_net, pinch_positions, parameters, validation_batch_size, no_users, device):
    policy_net.eval()
    x_dummy = make_dummy_x(n_antennas=n_antennas, n_users=n_users, dev=device)
    with torch.no_grad():

        #AYTO THA HTAN AN EKPAIDEYAME ME PINCH POS AND USER POS
        user_positions_val,B_val_polar, B_val, optimal_rates_val,SNRs_val, a_opts_val = preprocess_data(dataset_path=validation_path, total_samples=validation_batch_size,device = dev)
        user_pos_val = torch.from_numpy(user_positions_val).unsqueeze(1).to(torch.float32).to(device)
        #print("Validation Beta's: ", B_val_polar[0])

        # Create Graph
        batch_graph_val = create_batch_graph(user_pos = user_pos_val, pinch_positions = pinch_positions, B = B_val, B_polar = B_val_polar, dev = dev)

        pi_val, imps_val = policy_net(batch_graph_val, parameters["N_PINCHES"], validation_batch_size)
        action_val = (pi_val > 0.5).int()

        model_rates_val, model_SNRs_val = calculate_rates(B_val, batch_size=validation_batch_size,
                                          a_opt=action_val,
                                          parameters=parameters)
        for i in range(validation_batch_size):
            model_rate = model_rates_val[i].item()
            optimal_rate = optimal_rates_val[i]
            #print(f"[VAL] Sample {i}: Model Rate={model_rate:.4f} | Optimal Rate={optimal_rate:.4f}")

        avg_val_ratio = np.mean([model_rates_val[i].item() / optimal_rates_val[i] for i in range(validation_batch_size)])

    return avg_val_ratio


# ========= Main ===========
if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(2)
    torch.autograd.set_detect_anomaly(True)

    n_users = 1
    n_antennas = 50
    batch_size = 1000
    total_samples = 5000
    lr = 1e-3
    iterations = 2500
    parameters = get_antenna_parameters(n_antennas, n_users)
    batch_size_test = 100
    validation_samples=1000

    train_dataset_path = f"data/train/augmented_dataset_{total_samples}samples_{n_antennas}ant_SQUARE100_waveguide50.json"
    validation_path = f"data/val/augmented_dataset_{validation_samples}samples_{n_antennas}ant_SQUARE100_waveguide50.json"


    #checkpoint_path = f"checkpoint_GNN+MLP_{n_antennas}ant_{n_users}user_lr_{lr}_BCE_{total_samples}_samples_SQUARE100_waveguide50_hid64+128.pth"
    checkpoint_path = f"dummy.pth"
    
    pinch_positions = generate_normalized_pinch_positions(parameters)
    #pinch_positions = np.array([(i * 0.5, 0, 5) for i in range(n_antennas)])

    user_positions, B_polar, B, optimal_rates,SNRs, a_opts = preprocess_data(dataset_path=train_dataset_path, total_samples=total_samples, device=dev)
    print("A_opts[0]: ", a_opts[0])

    #policy = Policy_GNN_MLP(in_chnl=3, hid_chnl=64, mlp_hidden_dim=128, dev=dev).to(dev)
    #policy = Policy_GNN_MLP( in_chnl=1,  hid_chnl=16, mlp_hidden_dim=64, dev=dev)
    policy = Policy_GNN_MLP( in_chnl=1,  hid_chnl=64, mlp_hidden_dim=128, dev=dev)



    model, pi, best_results, result_list = train(optimal_rates, a_opts, parameters, total_samples, n_antennas, user_positions,
                                                 pinch_positions, B, policy, lr, n_users, iterations, dev, batch_size,
                                                 validation_path, checkpoint_path=checkpoint_path, validation_batch_size=validation_samples)

    #test_model(model, test_dataset_path, n_antennas, n_users, pinch_positions, parameters, batch_size_test, dev=dev)
