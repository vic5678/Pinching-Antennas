import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops, scatter

class CMPNNConv(MessagePassing):
    def __init__(self, edge_in_channels, out_channels):
        super(CMPNNConv, self).__init__(aggr='mean')  # or 'add', 'max'

        # MLP to transform edge features [B_mag, B_phase]
        self.edge_mlp = nn.Sequential(
            nn.Linear(2, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.edge_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self, edge_index, edge_attr, num_nodes):
        """
        edge_attr: shape [num_edges, edge_in_channels]
        edge_index: shape [2, num_edges]
        """
        return self.propagate(edge_index=edge_index, edge_attr=edge_attr, size=(num_nodes, num_nodes))

    def message(self, edge_attr):
        return self.edge_mlp(edge_attr)

    def update(self, aggr_out):
        return F.relu(aggr_out)


    
class Net(nn.Module):
    def __init__(self, in_chnl, hid_chnl):
        super(Net, self).__init__()

        # ✅ Initial projection layer (single transformation now)
        self.lin1 = nn.Linear(in_chnl, hid_chnl)
        self.bn1 = nn.BatchNorm1d(hid_chnl)
        #self.bn1 = nn.LayerNorm(hid_chnl)

        # ✅ MPNN convolutional layers (now using only one feature space)
        self.conv1 = CMPNNConv(hid_chnl, hid_chnl)
        self.bn2 = nn.BatchNorm1d(hid_chnl)  
        self.conv2 = CMPNNConv(hid_chnl, hid_chnl)
        self.bn3 = nn.BatchNorm1d(hid_chnl)
        # Add 3rd conv layer
        #self.conv3 = CMPNNConv(hid_chnl, hid_chnl)
        #self.bn4 = nn.BatchNorm1d(hid_chnl)
        #self.conv4 = CMPNNConv(hid_chnl, hid_chnl)
        #self.bn5 = nn.BatchNorm1d(hid_chnl)
        self.match_proj_conv2 = nn.Linear(hid_chnl, hid_chnl)
        self.match_proj_conv3 = nn.Linear(hid_chnl, hid_chnl)
        self.match_proj_conv4 = nn.Linear(hid_chnl, hid_chnl)

        # ✅ Graph pooling layers
        self.linears_prediction = nn.ModuleList()
        for layer in range(1+2):  # 1 initial + 2 convolution layers
            self.linears_prediction.append(nn.Linear(2*hid_chnl, hid_chnl)) ## The 2* is for both global and mean pooling

    def forward(self, x, edge_index, edge_attr, batch):
        # Just keep x as dummy zero input if unused
        num_nodes = x.size(0) if x is not None else int(edge_index.max()) + 1
        x = torch.zeros((num_nodes, self.lin1.in_features), device=edge_attr.device)

        h = F.relu(self.bn1(self.lin1(x)))  # Optionally remove if you don't need x at all

        hidden_rep = [h]

        h = F.relu(self.bn2(self.conv1(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)))
        hidden_rep.append(h)

        h = F.relu(self.bn3(self.conv2(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)))
        hidden_rep.append(h)

        node_pool_over_layer = sum(hidden_rep)

        gPool_over_layer = 0
        for layer, layer_h in enumerate(hidden_rep):
            g_pool = torch.cat([
                global_max_pool(layer_h, batch),
                global_mean_pool(layer_h, batch)
            ], dim=1)
            gPool_over_layer += F.dropout(self.linears_prediction[layer](g_pool), 0.5, training=self.training)

        return node_pool_over_layer, gPool_over_layer

