import gurobipy as gp
from gurobipy import GRB
import numpy as np
from torch_geometric.data import Data, Batch
import time
import torch
from policy_BCE import Policy
from policy_GNN_MLP import Policy_GNN_MLP
from MLPpolicy import MLPPolicy
from math import log2
from System_model_setup import calculate_rates, generate_normalized_pinch_positions, get_antenna_parameters, load_dataset, preprocess_data





def model_predictions(policy_test, test_size, test_dataset_path, n_antennas_test, n_users, pinch_positions, parameters, dev):
    # Load test data
    user_positions_test, B_polar_test, B_test, optimal_rates_test, _ = preprocess_data(
        dataset_path=test_dataset_path,
        total_samples=test_size,
        device=dev
    )

    user_pos_test = torch.from_numpy(user_positions_test).unsqueeze(1).to(torch.float32).to(dev)

    # === Construct graph data ===
    data_list = []
    adj = torch.zeros((test_size, n_users + n_antennas_test, n_users + n_antennas_test), dtype=torch.complex64)

    for b in range(test_size):
        for j in range(n_users):
            for i in range(1, n_antennas_test + 1):
                adj[b, j, i] = B_test[b, i - 1, j]

        edge_index = torch.nonzero(adj[b].abs() > 0).t()
        edge_attr = B_polar_test[b, :, 0, :]
        graph_x = torch.cat([user_pos_test[b], torch.from_numpy(pinch_positions)], dim=0)

        data_list.append(Data(x=graph_x, edge_index=edge_index, edge_attr=edge_attr))

    batch_graph_test = Batch.from_data_list(data_list).to(dev)

    # === Model prediction ===
    policy_test.eval()
    with torch.no_grad():
        # FOR GNN+DispN and GNN+MLP
        pi_test = policy_test(batch_graph_test, parameters["N_PINCHES"], test_size)
        # FOR MLP 
        #pi_test = policy_test(B_polar_test)
        action_test = (pi_test > 0.5).int()

    model_rates_test = calculate_rates(
        B=B_test,
        batch_size=test_size,
        a_opt=action_test,
        parameters=parameters
    )

    
    predictions = [action.cpu().numpy() for action in action_test]
    rates = [rate.item() for rate in model_rates_test]

    return predictions, rates

def solve_qf01p_unit_d(Q, tol=1e-4, max_iters=100, timeLimit = 10, Output_Flag = 1, k = None):
    n = Q.shape[0]
    model = gp.Model("QF01P_unit_d")
    model.setParam("OutputFlag", Output_Flag)
    model.setParam("TimeLimit", timeLimit)
    ### Early stopping
    model.setParam("MIPGap", 0.01)  

    # Binary decision variables
    x = model.addVars(n, vtype=GRB.BINARY, name="x")

    # Initial lambda
    lam = 0.0
    for iteration in range(max_iters):
        model.setObjective(0, GRB.MAXIMIZE)

        quad_expr = gp.QuadExpr()
        for i in range(n):
            for j in range(n):
                quad_expr.add(x[i] * x[j] * Q[i, j])
        lin_expr = gp.LinExpr(gp.quicksum(x[i] for i in range(n)))
        model.setObjective(quad_expr - lam * lin_expr, GRB.MAXIMIZE)
        model.addConstr(gp.quicksum(x[i] for i in range(n)) >= 1, name="nonzero_cardinality")
        if k is not None:
            model.addConstr(gp.quicksum(x[i] for i in range(n)) ==  int(round(k)),  name="cardinality_k")


        model.optimize()

        x_sol = np.array([x[i].X for i in range(n)])
        num = x_sol @ Q @ x_sol
        denom = np.sum(x_sol)

        new_lam = num / denom

    

        if abs(new_lam - lam) < tol:
            break
        lam = new_lam

    return x_sol, num, denom, num / denom

if __name__ == '__main__':

    n_antennas = 50
    n_users = 1
    n_samples = 1
    timeLimit = 3600
    lr = 1e-3
    dataset_path =  f"data/train/augmented_dataset_5000samples_50ant_SQUARE100_waveguide50.json"
    Rates_gurobi = np.zeros(n_samples)
    Sol_gurobi = np.zeros((n_samples, n_antennas), dtype=int)

    ### GUROBI ###
    if dataset_path:
        
        # === Load dataset once ===
        user_positions, B_all, optimal_rates,SNRs, a_opts = load_dataset(dataset_path, n_samples)
        counter = 0 
        total_start = time.time()  
        for i in range(len(B_all)):
            B_sample = B_all[i]  
            B_vec = B_sample.flatten().cpu().numpy().reshape(-1, 1)  
            Q = np.real(np.outer(B_vec, B_vec.conj()))  
            k = int(a_opts[0].sum().item() * 0.1)
            x_sol, num, denom, obj = solve_qf01p_unit_d(Q, Output_Flag=0, k=k)
            Rates_gurobi[i] = log2(1+10000*obj)
            Sol_gurobi[i] = x_sol
            print(f"GUROBI Solutions for sample {i}: {x_sol}" )
        total_end = time.time()  
        print(f"\n🔥 Total running time: {total_end - total_start:.2f} seconds")
        print("GUROBI RATES's': ", Rates_gurobi )
        print("GUROBI Solutions': ", Sol_gurobi )
       
        
   

   ### MODEL ###
    """
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'

        # GNN+DispN
        #policy = Policy(in_chnl=3, hid_chnl=16, n_users=n_users, key_size_embd=16,key_size_policy=16, val_size=16, clipping=10, dev=dev).to(dev)
        #model_path = f"checkpoint_{n_antennas_trained}ant_{n_users}user_lr_{lr}_BCE_5000_samples_BRUTE_z==0.pth"
    
        # MLP
        #policy = MLPPolicy(input_dim=2, hidden_dim=16, global_dim=16)
        #model_path = f"checkpoint_MLP_{n_antennas_trained}ant_{n_users}user_lr_{lr}_BCE_5000_samples_BRUTE_z==0.pth"

        # GNN+MLP
        policy = Policy_GNN_MLP(in_chnl=3,  hid_chnl=16, mlp_hidden_dim=64, dev=dev)
        model_path = f"checkpoint_50ant_{n_users}user_lr_{lr}_BCE_5000_samples_BRUTE_z==0_GNN+MLP.pth"

        checkpoint = torch.load(model_path, map_location=dev, weights_only=False)
        policy.load_state_dict(checkpoint["model_state_dict"])
        policy.eval()
        parameters_test = get_antenna_parameters(n_antennas, n_users)
        pinch_positions_test = generate_normalized_pinch_positions(parameters=parameters_test)
        Model_preds, Model_rates  = model_predictions(test_size=n_samples, test_dataset_path=dataset_path, policy_test=policy, pinch_positions=pinch_positions_test, n_antennas_test=n_antennas, parameters=parameters_test, n_users=n_users, dev=dev)
        print("MODEL PREDICTIONS: ", Model_preds)
        print("MODEL RATES: ", Model_rates)
    else:
        np.random.seed(0)
        n = n_antennas
        Q = np.random.randn(n, n)
        Q = (Q + Q.T) / 2  # Ensure symmetry
        total_start = time.time()
        x_sol, num, denom, obj = solve_qf01p_unit_d(Q, timeLimit=timeLimit)
        total_end = time.time()  # ⏱️ End timing
        print(f"\n🔥 Total running time: {total_end - total_start:.2f} seconds")
        print("Optimal x:", x_sol)
        print("Objective value:", obj)

    """

