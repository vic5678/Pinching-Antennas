import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import os
import torch
from torch_geometric.data import Data, Batch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F



def calculate_antenna_activation(actions):
    """
    Calculates average antenna activation metrics from a binary action array.

    Parameters:
        actions (np.ndarray): A NumPy array of shape [num_samples, n_users, n_antennas],
                              containing binary antenna selections (0 or 1).

    Returns:
        avg_active_per_sample (float): Average number of antennas activated per sample.
        avg_activation_fraction (float): Fraction of antennas activated overall (0 to 1).
    """
    total_active = np.sum(actions)
    total_possible = np.prod(actions.shape)

    avg_activation_fraction = total_active / total_possible
    avg_active_per_sample = total_active / actions.shape[0]

    return avg_active_per_sample, avg_activation_fraction


#### LOAD DATASET FOR NOISE TRAINING

def plot_from_checkpoint(checkpoint_path, validate_every=1, val_stride=5):
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint file not found: {checkpoint_path}")
        return

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    training_losses = checkpoint.get('training_losses', [])
    training_accuracies = checkpoint.get('training_accuracies', [])
    validation_loss_list = checkpoint.get('validation_loss_list', [])
    validation_accuracies = checkpoint.get('validation_accuracies', [])

    print("Training losses: ", training_losses[:5], "...")
    print("Training accuracies: ", training_accuracies[:5], "...")
    print("Validation losses: ", validation_loss_list[:5], "...")
    print("Validation accuracies: ", validation_accuracies[:5], "...")

    if not training_losses or not training_accuracies:
        print("❌ Missing required data in checkpoint.")
        return

    # X-axis for full training and validation
    train_iters = list(range(1, len(training_losses) + 1))
    val_iters = list(range(1, len(validation_loss_list) + 1))

    # Subsample validation losses and accuracies
    val_indices = list(range(0, len(validation_loss_list), val_stride))
    val_iters_sub = [val_iters[i] for i in val_indices]
    val_loss_sub = [validation_loss_list[i] for i in val_indices]
    val_acc_sub = [validation_accuracies[i] for i in val_indices]

    # --- Plot BCE Loss ---
    plt.figure(figsize=(10, 5))
    plt.plot(train_iters, training_losses, label='Training BCE Loss (per iteration)', alpha=0.5)
    plt.plot(val_iters_sub, val_loss_sub, label=f'Validation BCE Loss (every {val_stride} iters)',  color='orange')
    plt.xlabel("Iteration")
    plt.ylabel("BCE Loss")
    plt.title("Training vs Validation BCE Loss (subsampled)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Plot Accuracies ---
    plt.figure(figsize=(10, 5))
    plt.plot(train_iters, training_accuracies, label='Training Accuracy (rate ratio)', alpha=0.7)
    plt.plot(val_iters_sub, val_acc_sub, label=f'Validation Accuracy (every {val_stride} iters)', color='green')
    plt.xlabel("Iteration")
    plt.ylabel("Rate Ratio")
    plt.title("Training vs Validation Accuracy (Rate Ratio, subsampled)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def create_batch_graph(user_pos, pinch_positions, B, B_polar, dev):

    batch_size = user_pos.size(0)
    n_antennas = pinch_positions.shape[0]


    # Node features
    pinch_tensor = torch.from_numpy(pinch_positions).float().unsqueeze(0).repeat(batch_size, 1, 1).to(dev)

    fea_test = torch.cat([user_pos, pinch_tensor], dim=1)  # [B, 1 + n_antennas, 3]

    data_list = []

    # Create fixed edge_index: edges from user (0) to each antenna (1 to N)
    edge_index = torch.stack([
        torch.zeros(n_antennas, dtype=torch.long),                 # source = user (0)
        torch.arange(1, n_antennas + 1, dtype=torch.long)          # target = antennas
    ], dim=0).to(dev)
    
    
    for b in range(batch_size):
        edge_attr = B_polar[b, :, 0, :].to(dev)  # [n_antennas, edge_dim]
        data = Data(x=fea_test[b], edge_index=edge_index, edge_attr=edge_attr)
        data_list.append(data)

    batch_graph = Batch.from_data_list(data_list).to(dev)
    return batch_graph


def create_undirected_batch_graph(user_pos, pinch_positions, B_polar, dev):
    batch_size = user_pos.size(0)
    n_antennas = pinch_positions.shape[0]

    # Node features: [B, 1 + n_antennas, 3]
    pinch_tensor = torch.from_numpy(pinch_positions).float().unsqueeze(0).repeat(batch_size, 1, 1).to(dev)
    fea_test = torch.cat([user_pos, pinch_tensor], dim=1)

    data_list = []

    # Create undirected edges: user (0) ↔ each antenna (1 to N)
    user_to_antennas = torch.stack([
        torch.zeros(n_antennas, dtype=torch.long),
        torch.arange(1, n_antennas + 1, dtype=torch.long)
    ], dim=0)
    antennas_to_user = torch.stack([
        torch.arange(1, n_antennas + 1, dtype=torch.long),
        torch.zeros(n_antennas, dtype=torch.long)
    ], dim=0)

    edge_index = torch.cat([user_to_antennas, antennas_to_user], dim=1).to(dev)  # shape: [2, 2*n_antennas]

    for b in range(batch_size):
        # B_polar[b, :, 0, :] gives shape [n_antennas, 2] (user → antenna)
        edge_attr = torch.cat([
            B_polar[b, :, 0, :],  # user → antenna
            B_polar[b, :, 0, :]   # antenna → user (use same features)
        ], dim=0).to(dev)  # shape: [2*n_antennas, 2]

        data = Data(x=fea_test[b], edge_index=edge_index, edge_attr=edge_attr)
        data_list.append(data)

    batch_graph = Batch.from_data_list(data_list).to(dev)
    return batch_graph



def generate_normalized_pinch_positions(parameters):
    
    N = parameters["N_PINCHES"]
    H = parameters["H"]
    square_side = parameters["SQUARE_SIDE"]

    # Generate equally spaced antenna positions along the x-axis
    pinch_positions = np.array([
        (-square_side/2 + i * square_side / N, 0 , H) for i in range(N)
    ])
    return pinch_positions

'''
def get_antenna_parameters(no_antennas, no_users):
    """
    Returns:
        dict: Dictionary containing system model parameters.
    """
    c = 3e8  # speed of light (m/s)
    fc = 28e9  # carrier frequency: 28 GHz
    lam = c / fc  # free-space wavelength
    neff = 1.4  # effective refractive index
    lam_g = lam / neff  # guided wavelength

    eta = c**2 / (16 * np.pi**2 * fc**2)  # from Ding's model

    parameters = {
        "N_PINCHES": no_antennas,      # Number of antennas
        "n_users": no_users,           # Number of users
        "P": 0.001,                      # Transmit power in Watts (20 dBm)
        "N": no_antennas,              # Total number of antennas
        "LAMBDA": lam,                 # Free-space wavelength
        "LAMBDA_G": lam_g,             # Guided wavelength
        "DISTANCE": 0.5,               # Distance between pinches (m)
        "ETA": eta,                    # Aperture gain factor
        "SIGMA2": 1e-12,                # Noise power
        "H": 3,                      # Antenna height (m)
        "SQUARE_SIDE": 10.0,           # Room/array extent (m)
        "PSI_0": np.array([-5.0, 0.0, 3])  # Waveguide feed center
    }
    

    return parameters
'''

def get_antenna_parameters(no_antennas, no_users):
    """
    Returns:
        dict: Dictionary containing all parameters.
    """
    parameters = {
        "N_PINCHES": no_antennas,  # Number of pinches
        "n_users": no_users,  # Number of users
        "P": 1,  # Total Power (normalized)
        "N": no_antennas,  # Since power is divided equally
        "LAMBDA_G": 10,  # Waveguide wavelength
        "LAMBDA": 1,  # Free-space wavelength (normalized)
        "DISTANCE": 1/2,  # Distance between pinches
        "ETA": 1,  # Path loss factor
        "SIGMA2": 0.0001,  # Noise power at user
        "H": 3,  # Height of pinches
        "SQUARE_SIDE": 50.0,
        "PSI_0": np.array([-50/2, 0, 3])  # Feed point of waveguide (assumed at origin)
        
    }
    
    return parameters

def beta_calc(N_PINCHES, pinch_positions, user_positions, LAMBDA, LAMBDA_G, PSI_0, batch_size):
    
    n_users = 1
    beta_n = np.zeros((batch_size, N_PINCHES, n_users), dtype=complex) 
    
    
    for b in range(batch_size):  # Iterate over batch
        for j in range (n_users):
            for i in range(N_PINCHES):
                x_n, y_n, z_n = pinch_positions[i]
                psi_n = np.array([x_n, y_n, z_n])
                
                # Compute θ(ψ_n) - Phase shift due to position
                theta_n = (2 * np.pi / LAMBDA_G) * np.linalg.norm(PSI_0 - psi_n)
                
                # Compute 3D distance for each user in batch
                distance = np.linalg.norm(user_positions[b] - psi_n)

                # Compute phase shift (propagation + intrinsic)
                phase_shift = np.exp(-1j * ((2 * np.pi * distance / LAMBDA) + theta_n))
                
                beta_n[b, i , j] = phase_shift / distance  # Store values for batch index b
    #print("Beta shape inside beta_calc: ", beta_n.shape)
    return beta_n

def calculate_rates_torch(B,batch_size, a_opt, parameters):
    """
    Differentiable version of calculate_rates using PyTorch.

    Args:
        B: [B, N, 1] complex tensor
        a_opt: [B, 1, N] float or binary (0/1) tensor
        parameters: dict of system constants

    Returns:
        rate: [B] tensor
        SNR: [B] tensor
    """
    P = parameters["P"]
    ETA = parameters["ETA"]
    SIGMA2 = parameters["SIGMA2"]

    # B: [B, N, 1]  -> squeeze to [B, N]
    B = B.squeeze(-1)

    # a_opt: [B, 1, N] → transpose to [B, N] to match B
    a_opt = a_opt.squeeze(1)

    # Select antenna contributions: zero out inactive
    selected_B = B * a_opt  # [B, N] complex

    # Sum signals across antennas
    received_signal = torch.sum(torch.sqrt(torch.tensor(ETA, device=B.device)) * selected_B, dim=1)  # [B]

    # Power
    received_power = torch.abs(received_signal) ** 2  # [B]

    # Count active antennas
    N_active = a_opt.sum(dim=1)  # [B]

    # Avoid divide-by-zero: mask for samples with N_active > 0
    mask = N_active > 0
    SNR = torch.zeros_like(received_power)
    rate = torch.zeros_like(received_power)

    # Compute SNR only where valid
    SNR[mask] = P * ETA *received_power[mask] / (SIGMA2 * N_active[mask])
    rate[mask] = torch.log2(1 + SNR[mask])

    return rate, SNR


def calculate_rates(B, batch_size, a_opt, parameters):
    # Unpack parameters
    LAMBDA = parameters["LAMBDA"]
    LAMBDA_G = parameters["LAMBDA_G"]
    PSI_0 = parameters["PSI_0"]
    P = parameters["P"]
    ETA = parameters["ETA"]
    SIGMA2 = parameters["SIGMA2"]
    N_PINCHES = parameters["N_PINCHES"]
    n_users = 1

    assert B.shape[0] == a_opt.shape[0], f"Batch size mismatch: B {B.shape[0]} vs a_opt {a_opt.shape[0]}"


    # Ensure inputs are NumPy arrays
    B = np.array(B, dtype=np.complex128)
    a_opt = a_opt.detach().cpu().numpy()


    # Select only the antennas that are active (a_opt == 1)
    # Shape of B: (batch_size, N_PINCHES, n_users)
    # Shape of a_opt: (batch_size, n_users, N_PINCHES)
    # We transpose a_opt to match B's layout
    a_opt_transposed = np.transpose(a_opt, (0, 2, 1))  # (batch_size, N_PINCHES, n_users)

    # Multiply by a_opt to zero out inactive antennas
    selected_B = B * a_opt_transposed  # broadcasting happens here

    # Sum the complex signal contributions for each batch
    received_signal = np.sum(np.sqrt(ETA) * selected_B, axis=1)  # shape: (batch_size, n_users)

    # Received power: |signal|^2
    received_power = np.abs(received_signal) ** 2  # shape: (batch_size, n_users)

    # Count active antennas per sample
    N_active_antennas = np.sum(a_opt, axis=2)  # shape: (batch_size, n_users)

    # Avoid division by zero by masking
    mask = N_active_antennas != 0
    SNR = np.zeros_like(received_power)
    rate = np.zeros_like(received_power)

    # Calculate SNR and rate only where antennas are active
    SNR[mask] = P * received_power[mask] / (SIGMA2 * N_active_antennas[mask])
    rate[mask] = np.log2(1 + SNR[mask])

    # Remove singleton user dimension if n_users == 1
    return rate.squeeze(), SNR.squeeze()

"""
def calculate_rates(B , batch_size ,  a_opt, parameters ):
    
    # Unpack parameters
    LAMBDA = parameters["LAMBDA"]
    LAMBDA_G = parameters["LAMBDA_G"]
    PSI_0 = parameters["PSI_0"]
    P = parameters["P"]
    ETA = parameters["ETA"]
    SIGMA2 = parameters["SIGMA2"]
    N_PINCHES = parameters["N_PINCHES"]
    n_users = 1

    # 🔹 Compute received signal using selected antennas 
    received_signal = np.zeros(batch_size,dtype=np.complex64) 
    received_power = np.zeros(batch_size)
    rate = np.zeros(batch_size)
    SNR = np.zeros(batch_size)
    #print("inside calculate rates: B.shape: ", B.shape)
    for b in range(batch_size):  # Loop over batch
        for j in  range(n_users):
            for i in range(N_PINCHES):  # Loop over antennas
                if a_opt[b,j,i].item() != 0 :
                    
                    #received_signal[b] += np.complex64(np.sqrt(P / N_PINCHES) * np.sqrt(ETA) * B[b, i, j].item()/10)
                    received_signal[b] += np.complex128(np.sqrt(ETA) * B[b, i, j].item())
                    #received_signal[b] += np.sqrt(ETA) * B[b, i, j].cpu().numpy().astype(np.complex128)

                    #print("received_signal[b]: ", received_signal[b])
                    #print("B[b,i,j] = ", B[b,i,j])
        #N_active_antennas = int(torch.sum(a_opt[b][j]).item())  # ✅ or use .sum().item()
        N_active_antennas = sum(a_opt[b][j])
        received_power[b] = np.abs(received_signal[b]) ** 2
        if N_active_antennas==0:
           rate[b]=0
           SNR[b]=0
        else:
           SNR[b] = P* received_power[b] / (SIGMA2 * N_active_antennas)
           rate[b] = np.log2(1 + SNR[b] )
        

    return rate, SNR
"""
def loss_with_soft_gamma(imps, target_actions, B_complex, B_numpy, SNR_opt, parameters,batch_size, pos_weight_value=1.6, lambda_gamma=1.0, lambda_collapse = 10.0):
    """
    Combines BCEWithLogitsLoss with a differentiable gamma penalty using soft activations.

    Args:
        imps: [B, N] logits from the model
        target_actions: [B, N] float labels (0/1)
        B_complex: [B, N] complex64 channel matrix
        parameters: dict with "THETA", "P_TX", "ETA", "NOISE"
        pos_weight_value: scalar for BCE class balance
        lambda_gamma: scalar weight for gamma penalty

    Returns:
        total_loss, bce_loss, gamma_penalty, gamma_soft.detach()
    """
    device = imps.device
    

    # BCE loss on logits
    pos_weight = torch.tensor(pos_weight_value, device=device)
    bce_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bce_loss = bce_fn(imps, target_actions)
    #print("BCE loss with weights: ", bce_loss)
    # Soft activations
    soft_actions = torch.sigmoid(imps)  # [B, N]

    _, gamma_soft_np = calculate_rates_torch(B_numpy, batch_size, soft_actions, parameters)

    # Compute gamma ratio
    SNR_opt = torch.tensor(SNR_opt, dtype=torch.float32, device=imps.device)
    gamma_ratio = gamma_soft_np / (SNR_opt)
    #print("gamma ratio = ", gamma_ratio)
    #gamma_ratio = torch.tensor(gamma_ratio, dtype=torch.float32, device=imps.device)
    gamma_ratio = torch.minimum(gamma_ratio, torch.ones_like(gamma_ratio))
    
    
    # Gamma ratio penalty
    gamma_ratio_loss = ((1 - gamma_ratio) ** 2).mean()


    

    # Collapse penalty for low gamma_soft
    collapse_penalty = F.relu(0.1 - gamma_ratio).mean()
    #print("collapse_penalty = ", collapse_penalty)

    # Total loss
    total_loss = 0.3*bce_loss + lambda_gamma * gamma_ratio_loss + lambda_collapse * collapse_penalty
    #total_loss = lambda_gamma * gamma_ratio_loss 
    #total_loss = bce_loss + lambda_gamma * gamma_penalty
    
    #gamma_soft = gamma_soft / (SNR_opt + 1e-8)
    #SNR_opt = torch.ones_like(gamma_soft)  # target ratio = 1
    #mse_loss = F.mse_loss(gamma_soft/SNR_opt, 1)
    #print("mse loss: ", mse_loss)
 
    #print("MSE loss: ", mse_loss)
    #total_loss = 0.5*bce_loss + 2*mse_loss
    
    #print("Final loss: ", total_loss)

    return total_loss

def preprocess_data(dataset_path, total_samples, device, use_noisy_B=False):
    # Load raw data
    user_positions, B, B_noisy,optimal_rates, SNRs, a_opts = load_dataset(
        dataset_path, total_samples, device=device, use_noisy_B=use_noisy_B
    )

    # Compute magnitude & phase
    B_mag = torch.abs(B).unsqueeze(-1)
    B_phase = torch.angle(B).unsqueeze(-1)
    B_polar = torch.cat([B_mag, B_phase], dim=-1)

    return user_positions, B_polar, B, B_noisy, optimal_rates, SNRs, a_opts



def load_dataset(dataset_path, batch_size=1, device='cpu', use_noisy_B=False):
    """
    Loads user positions and complex B from JSON dataset,
    returns:
    - user_positions: numpy array [batch_size, 3]
    - B: torch.ComplexFloatTensor [batch_size, N_PINCHES, n_users]
    - optimal_rates: list of floats [batch_size]
    """
    # Load dataset
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # Select first `batch_size` samples
    selected_samples = dataset[:batch_size]

    # Extract user positions
    user_positions = np.array([sample["user_position"] for sample in selected_samples])  # [batch_size, 3]

    # Extract complex B (stored as list of complex numbers)
    ##### FOR NOISE TRAINING ######
    if use_noisy_B == True:
        B_list = [sample["B"] for sample in selected_samples]  # List of lists
        B_list1 = [sample["B_noisy"] for sample in selected_samples]  # List of lists
    else:
        B_list = [sample["B"] for sample in selected_samples]  # List of lists
        B_list1=[]
    '''
    if use_noisy_B == False:
        B_list = [sample["B"] for sample in selected_samples]  # List of lists
    else:
        B_list = [sample["B_noisy"] for sample in selected_samples]  # List of lists
    '''
    # Convert list of lists into numpy array of complex numbers
    B_complex_np = np.array(B_list, dtype=np.complex64)  # [batch_size, N_PINCHES, n_users]
    B_complex_np1 = np.array(B_list1, dtype=np.complex64)  # [batch_size, N_PINCHES, n_users]

    # Convert to torch tensor
    B = torch.from_numpy(B_complex_np).to(torch.complex64).to(device)
    B_noisy = torch.from_numpy(B_complex_np1).to(torch.complex64).to(device)

    # Extract optimal rates
    optimal_rates = [sample["optimal_rate"] for sample in selected_samples]

    # Extract SNRs
    SNRs = [sample["SNR"] for sample in selected_samples]

    # Extract a_opt values
    a_opts = np.array([sample["a_opt"] for sample in selected_samples])  # [batch_size, N_PINCHES]



    return user_positions, B, B_noisy, optimal_rates, SNRs , a_opts

def plot_antenna_selection(user_pos, pinch_positions, selected_antennas, title="Antenna Selection"):
    """
    3D visualization of user and selected antennas.

    Args:
        user_pos: numpy array of shape [3]
        pinch_positions: numpy array of shape [n_antennas, 3]
        selected_antennas: numpy array of shape [n_antennas], 0 or 1
        title: plot title
    """
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot all antennas
    ax.scatter(pinch_positions[:, 0], pinch_positions[:, 1], pinch_positions[:, 2],
               c='gray', label='All Antennas', alpha=0.3)

    # Selected antennas
    selected_idxs = np.where(selected_antennas == 1)[0]
    ax.scatter(pinch_positions[selected_idxs, 0],
               pinch_positions[selected_idxs, 1],
               pinch_positions[selected_idxs, 2],
               c='green', label='Open Antennas', s=60)

    # User
    ax.scatter(user_pos[0], user_pos[1], user_pos[2], c='red', label='User', s=100, marker='X')

    # Lines from antennas to user
    for i in selected_idxs:
        ax.plot([pinch_positions[i, 0], user_pos[0]],
                [pinch_positions[i, 1], user_pos[1]],
                [pinch_positions[i, 2], user_pos[2]],
                c='blue', alpha=0.3, linestyle='--')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()