from EdgeBasedCmpnn import Net
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
import torch
import math
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.data import Data
from torch_geometric.data import Batch
from System_model_setup import plot_antenna_selection, get_antenna_parameters, beta_calc, create_batch_graph, load_dataset

class Policy_GNN_MLP(nn.Module):
    def __init__(self, in_chnl, hid_chnl, mlp_hidden_dim, dev):
        super(Policy_GNN_MLP, self).__init__()
        self.dev = dev

        self.gnn = Net(in_chnl=in_chnl, hid_chnl=hid_chnl).to(dev)
        self.mlp = nn.Sequential(
            nn.Linear(hid_chnl * 2, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, batch_graph, n_antennas, n_batch):
        nodes_h, g_h = self.gnn(
            x=batch_graph.x.to(torch.float32),
            edge_index=batch_graph.edge_index,
            edge_attr=batch_graph.edge_attr,
            batch=batch_graph.batch
        )

       
        nodes_h = nodes_h.view(n_batch, n_antennas + 1, -1)  # (batch, total_nodes, hidden)
        g_h = g_h.view(n_batch, 1, -1)  # (batch, 1, hidden)

        # Extract antenna embeddings
        antenna_h = nodes_h[:, 1:, :]  # (batch, n_antennas, hidden)

        g_h_repeated = g_h.repeat(1, n_antennas, 1)  # (batch, n_antennas, hidden)
        fusion_input = torch.cat([antenna_h, g_h_repeated], dim=-1)  # (batch, n_antennas, 2*hidden)
        fusion_input = fusion_input.view(-1, fusion_input.shape[-1])  # (batch * n_antennas, 2*hidden)

        #logits = self.mlp(fusion_input)  # (batch * n_antennas, 1)
        #prob = logits.view(n_batch, 1, n_antennas)
        logits = self.mlp(fusion_input)  # (batch * n_antennas, 1)
        #print("📢 Logits before sigmoid:", logits.view(n_batch, n_antennas)) 

        prob = self.sigmoid(logits).view(n_batch,1, n_antennas)  # final probabilities
        return prob , logits




    
if __name__ == '__main__':

    dev = 'cpu'
    torch.manual_seed(2)
    n_antennas = 50
    n_users = 1
    parameters = get_antenna_parameters(no_antennas=n_antennas, no_users=n_users)

    batch_size = 1

    dataset_path = f"data/test/augmented_dataset_1samples_50ant_SQUARE100_waveguide50.json"
    user_positions, B, optimal_rates,SNRs, a_opts = load_dataset(dataset_path, batch_size, device=dev)
    print("A_opts: " , a_opts) 

    ### ANTENNA POSITIONS ###
    pinch_positions = np.array([(i * parameters["DISTANCE"], 0, parameters["H"]) for i in range(parameters["N_PINCHES"])])
    user_pos = torch.from_numpy(user_positions).unsqueeze(1).to(torch.float32).to(dev)  # Shape: (batch_size, 1, 3)


    
    # Compute magnitude stats
    B_mag =  torch.abs(B)
    B_phase = torch.angle(B)


    B_mag = B_mag.unsqueeze(-1) 
    B_phase = B_phase.unsqueeze(-1)  


    print("B_mag after unsqueeze shape: ", B_mag.shape)  
    print("B_mag: ", B_mag) 
    print("B_phase after unsqueeze shape: ", B_phase.shape)  
    print("B_phase: ", B_phase)


    B_polar = torch.cat([B_mag, B_phase], dim=-1)
   
    print("B_polar shape: ", B_polar.shape)
    print("B_polar : ", B_polar)

  

    ### NODE FEATURES ###

    fea = torch.from_numpy(pinch_positions).unsqueeze(0).repeat(batch_size, 1, 1)
    fea = torch.cat([user_pos, fea], dim=1)

    print("fea shape: ", fea.shape)
    print("features: ", fea)

    # BETA calculation
    
    print("B shape: ", B.shape)  
    print("Betas: ", B)

     # Graph creation
    batch_graph = create_batch_graph(user_pos = user_pos, pinch_positions = pinch_positions, B = B, B_polar = B_polar, dev = dev)


    # Debugging prints
    print(f"🔹 Edge Index shape: {batch_graph.edge_index.shape}")
    print(f"🔹 Edge Attribute shape: {batch_graph.edge_attr.shape}")

    for i in range(batch_graph.edge_index.shape[1]):
        print(f"Edge {i}: {batch_graph.edge_index[:, i].tolist()} | Attr: {batch_graph.edge_attr[i].tolist()}")

    for g_id in range(batch_graph.batch.max().item() + 1):
        nodes_in_graph = (batch_graph.batch == g_id).nonzero(as_tuple=True)[0]
        print(f"Graph {g_id} Nodes: {nodes_in_graph.tolist()}")

    ### POLICY ###

    policy = Policy_GNN_MLP( in_chnl=1,  hid_chnl=64, mlp_hidden_dim=128, dev=dev)

    # Run policy forward
    pi, imps = policy(batch_graph, parameters["N_PINCHES"], batch_size)

    # Compute loss
    target_actions = torch.tensor(a_opts, dtype=torch.float32).to(pi.device).unsqueeze(1)
    loss = nn.BCELoss()(pi, target_actions)

    # Print debugging info
    print("Policy output (π):", pi)
    print("Target actions:", target_actions)
    print("BCE LOSS:", loss)

    # Optional: Compute gradient for analysis
    grad = torch.autograd.grad(pi.sum(), [param for param in policy.parameters()], allow_unused=True)

