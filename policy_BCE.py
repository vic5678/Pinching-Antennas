from EdgeBasedCmpnn_GNN_DispN import Net
#from cmpnn_polarBetas import Net
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
from System_model_setup import plot_antenna_selection, get_antenna_parameters, load_dataset, beta_calc, create_batch_graph, create_undirected_batch_graph, calculate_rates_torch, loss_with_soft_gamma


    

class Agentembedding(nn.Module):
    def __init__(self, g_h_size, depots_h_size, key_size, value_size, n_users):
        super(Agentembedding, self).__init__()
        self.key_size = key_size
       
        q_agent_input_size = g_h_size + depots_h_size  

        self.q_agent = nn.Linear(q_agent_input_size, key_size)
        self.k_agent = nn.Linear(depots_h_size // n_users, key_size)  
        self.v_agent = nn.Linear(depots_h_size // n_users, value_size)

    def forward(self, f_c, f):
        q = self.q_agent(f_c)
        k = self.k_agent(f)
        v = self.v_agent(f)
        u = torch.matmul(k, q.transpose(-1, -2)) / math.sqrt(self.key_size)

        ### u_ --> Attention Weights
        #u_ = entmax15(u, dim=-1)
        #u_ = sparsemax(u, dim=1).transpose(-1, -2)
        
        u_ = torch.sigmoid(u).transpose(-1, -2)

        #u_ = F.softmax(u, dim=-2).transpose(-1, -2)
        agent_embedding = torch.matmul(u_, v)

        return agent_embedding


class AgentAndNode_embedding(torch.nn.Module):
    def __init__(self, in_chnl, hid_chnl, n_users, key_size, value_size, dev):
        super(AgentAndNode_embedding, self).__init__()

        self.n_users = n_users

        # MPNN
        self.gin = Net(in_chnl=in_chnl, hid_chnl=hid_chnl).to(dev)

        # Depot's features size
        depots_h_size = self.n_users * hid_chnl  

        # Agent attention embedding
        self.agents = torch.nn.ModuleList()
        for i in range(n_users):
            self.agents.append(
                Agentembedding(
                    key_size=key_size,
                    value_size=value_size,
                    n_users=n_users,
                    g_h_size=hid_chnl,  
                    depots_h_size=depots_h_size  
                ).to(dev)
            )

    def forward(self, batch_graphs, n_antennas, n_batch):
        nodes_h, g_h = self.gin(
            x=batch_graphs.x.to(torch.float32),
            edge_index=batch_graphs.edge_index,
            edge_attr=batch_graphs.edge_attr,  
            batch=batch_graphs.batch
        )

        nodes_h = nodes_h.reshape(n_batch, n_antennas, -1)
        g_h = g_h.reshape(n_batch, 1, -1) 
      
        depots_h = nodes_h[:, :self.n_users, :] 
        depots_h_flattened = depots_h.reshape(n_batch, -1)  

       
        depot_cat_g = torch.cat((g_h.squeeze(1), depots_h_flattened), dim=-1).unsqueeze(1)  
       
       
        nodes_h_no_depot = nodes_h[:, self.n_users:, :]

       
    
        agents_embedding = []
        for i in range(self.n_users):
            agents_embedding.append(self.agents[i](depot_cat_g, nodes_h_no_depot))

        agent_embeddings = torch.cat(agents_embedding, dim=1)

        return agent_embeddings, nodes_h_no_depot




class Policy(nn.Module):
    def __init__(self, in_chnl, hid_chnl, n_users, key_size_embd, key_size_policy, val_size, clipping, dev):
        super(Policy, self).__init__()
        self.c = clipping
        self.key_size_policy = key_size_policy

        # Linear layers for key-value attention
        self.key_policy = nn.Linear(hid_chnl, self.key_size_policy).to(dev)
        self.q_policy = nn.Linear(val_size, self.key_size_policy).to(dev)
        #self.key_policy = nn.Sequential(nn.Linear(hid_chnl, 32),nn.ReLU(),nn.Linear(32, self.key_size_policy)).to(dev)

        #self.q_policy = nn.Sequential(nn.Linear(val_size, 32),nn.ReLU(),nn.Linear(32, self.key_size_policy)).to(dev)

        #self.key_policy = nn.Sequential(nn.Linear(2 * hid_chnl, 128),nn.ReLU(),nn.LayerNorm(128),nn.Linear(128, self.key_size_policy)).to(dev)
        #self.q_policy = nn.Sequential(nn.Linear(val_size, 128),nn.ReLU(),nn.LayerNorm(128),nn.Linear(128, self.key_size_policy)).to(dev)

        # Embedding network
        self.embed = AgentAndNode_embedding(
            in_chnl=in_chnl, hid_chnl=hid_chnl, n_users=n_users,
            key_size=key_size_embd, value_size=val_size, dev=dev
        )

    def forward(self, batch_graph, n_antennas, n_batch):
      
        agent_embeddings, nodes_h_no_depot = self.embed(batch_graph, n_antennas+1, n_batch)
        
        # Apply key and query transformation
        k_policy = self.key_policy(nodes_h_no_depot)
        q_policy = self.q_policy(agent_embeddings)

        # Compute attention scores
        u_policy = torch.matmul(q_policy, k_policy.transpose(-1, -2)) / math.sqrt(self.key_size_policy)
       
        imp = self.c * torch.tanh(u_policy)
        #print("IMPORTANCE: ", imp)
        
        prob = torch.sigmoid(imp)

        return prob , imp


def action_sample(pi):

    dist = Bernoulli(probs=pi) 
    #dist = Categorical(probs=pi) 

    action = dist.sample()  
    action = action.float() 
    action = action.to(torch.int)

    
    log_prob = dist.log_prob(action.float())  # Convert back to float for log_prob computation

    return action, log_prob



if __name__ == '__main__':
    

    dev = 'cpu'
    torch.manual_seed(2)
    n_antennas = 50
    n_users = 1
    parameters = get_antenna_parameters(no_antennas=n_antennas, no_users=n_users)

    batch_size = 1

    dataset_path = f"data/test/augmented_dataset_1samples_50ant_SQUARE100_waveguide50_NEWSNR.json"

    user_positions, B,B_noisy, optimal_rates,SNRs, a_opts = load_dataset(dataset_path, batch_size, device=dev)
    print("A_opts: " , a_opts) 

    
    pinch_positions = np.array([(i * parameters["DISTANCE"], 0, parameters["H"]) for i in range(parameters["N_PINCHES"])])
    user_pos = torch.from_numpy(user_positions).unsqueeze(1).to(torch.float32).to(dev)  

   
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
    print("B shape: ", B.shape)  
    print("Betas: ", B)

    
    #batch_graph = create_batch_graph(user_pos = user_pos, pinch_positions = pinch_positions, B = B, B_polar = B_polar, dev = dev)
    batch_graph = create_undirected_batch_graph(user_pos = user_pos, pinch_positions = pinch_positions, B_polar = B_polar, dev = dev)
    

  
    print(f"🔹 Edge Index shape: {batch_graph.edge_index.shape}")  
    print(f"🔹 Edge Index: {batch_graph.edge_index}")  
    print(f"🔹 Edge Attribute shape: {batch_graph.edge_attr.shape}")  
    print(f"🔹 Edge Attribute: {batch_graph.edge_attr}")  
 
    for i in range(batch_graph.edge_index.shape[1]):
        print(f"Edge: {batch_graph.edge_index[:, i].tolist()}, Attribute: {batch_graph.edge_attr[i].tolist()}")
    
    for g_id in range(batch_graph.batch.max().item() + 1):
        nodes_in_graph = (batch_graph.batch == g_id).nonzero(as_tuple=True)[0]
        print(f"Graph {g_id} Nodes: {nodes_in_graph.tolist()}")

    ### POLICY ###

    policy = Policy(in_chnl=3, hid_chnl=128, n_users=n_users, key_size_embd=64, key_size_policy=64, val_size=64, clipping=10, dev=dev).to(dev)
    pi , imps= policy(batch_graph, parameters["N_PINCHES"], batch_size)
    print("Policy: ", pi)
    #grad = torch.autograd.grad(pi.sum(), [param for param in policy.parameters()], allow_unused=True)
    target_actions = torch.tensor(a_opts, dtype=torch.float32).to(pi.device).unsqueeze(1)
    print("Target Actions: ", target_actions)
    print("a_opts: ",a_opts)
    BCE_loss = nn.BCELoss()(pi, target_actions)
    print("BCE LOSS: ", BCE_loss)

    loss = loss_with_soft_gamma(imps, target_actions, B_polar, B, SNRs, parameters,batch_size, pos_weight_value=1.6, lambda_gamma=1.0)
    print("Final hybrid LOSS = ", loss)




    