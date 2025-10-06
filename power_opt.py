import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math

# ==== your project imports (unchanged) ====
from policy_BCE import Policy                    # must return (prob, imp)
from System_model_setup import (
    beta_calc, preprocess_data, create_batch_graph,
    get_antenna_parameters, generate_normalized_pinch_positions
)

# -------------------------------------------------------------
# Utility: compute SNR and rate from B and a power vector p
# -------------------------------------------------------------
def snr_and_rate_with_power(B_row, p_row, noise_var=1.0):
    """
    SNR = |sum_n sqrt(p_n) * B_n|^2 / noise_var
    Rate = log2(1 + SNR)
    Inputs:
        B_row: complex tensor [N]
        p_row: real tensor    [N], >=0
    """
    s = (torch.sqrt(p_row.clamp_min(0)) * B_row).sum()
    snr = (s.abs() ** 2) / noise_var
    rate = torch.log2(1 + snr)
    return snr, rate


# -------------------------------------------------------------
# Phase-aware selection + power split (dominant phasor projection)
# -------------------------------------------------------------





@torch.no_grad()
def select_and_split_power_phase_aware(
    prob_K,           # [K, U, N] probabilities from K noisy user-position samples
    imp_K,            # [K, U, N] logits ("importance") from K noisy samples
    B_ref_row,        # complex [N] reference channel (e.g., mean over the K noisy Bs)
    P_tot=1.0,        # total power budget per user
    thr=0.5,          # selection threshold on mean probability
    eps=1e-6,         # tiny number for numerical stability
    topk_fallback=1,  # if none pass threshold, activate top-k by mean logits
    alpha_proj=1.0,   # exponent on projection magnitude
    beta_attn=1     # exponent on attention/importance
):
    """
    Phase-aware power split using dominant phasor projection + (optional) importance.
    Returns:
        final_action: [U, N] int {0,1}
        p_alloc:      [U, N] float, sums to P_tot over selected
        prob_mean:    [U, N]
        imp_mean:     [U, N]
    """
    # 1) Aggregate across K
    prob_mean = prob_K.mean(dim=0)  # [U, N]
    imp_mean  = imp_K.mean(dim=0)   # [U, N]
    U, N = prob_mean.shape

    # 2) Selection by threshold; ensure at least 1 per user
    sel_bool = (prob_mean > thr)    # [U, N] bool
    for u in range(U):
        if sel_bool[u].sum() == 0:
            topk = torch.topk(imp_mean[u], k=topk_fallback).indices
            sel_bool[u, topk] = True

    final_action = sel_bool.to(torch.int)                   # [U, N]
    p_alloc = torch.zeros((U, N), device=prob_mean.device, dtype=torch.float32)

    # Ensure reference channel is 1-D complex [N]
    B_ref_row = B_ref_row.reshape(-1).contiguous()

    P_tot_scalar = float(P_tot)

    for u in range(U):
        # ---- use integer indices (no boolean writes) ----
        idx = torch.nonzero(sel_bool[u], as_tuple=False).squeeze(1)  # [K_sel]
        K_sel = int(idx.numel())
        if K_sel == 0:
            continue

        # Selected entries
        B_sel  = B_ref_row.index_select(0, idx)                       # complex [K_sel]
        att_u  = imp_mean[u].index_select(0, idx).relu()              # [K_sel], >=0

        vec_sum = B_sel.sum()
        phi = torch.angle(vec_sum) if vec_sum.abs() > eps else torch.tensor(0.0, device=B_sel.device)

        # Positive real projection onto dominant direction
        proj = torch.real(B_sel * torch.exp(-1j * phi)).clamp(min=0.0)   # [K_sel] >= 0

        # Combined score (keep strictly 1-D)
        score = (proj + eps).pow(alpha_proj) * (att_u + eps).pow(beta_attn)
        score = score.reshape(-1).contiguous()

        if score.sum() <= eps:
            w_sel = torch.full_like(score, 1.0 / K_sel)
        else:
            w_sel = score / score.sum()                                  # [K_sel]

        # Full-length weights then scale to P_tot; write with integer index
        w_full = torch.zeros(N, device=prob_mean.device, dtype=torch.float32)
        w_full.index_copy_(0, idx, w_sel.to(torch.float32))              # safe write
        p_alloc[u] = w_full * P_tot_scalar

    return final_action, p_alloc, prob_mean, imp_mean

# -------------------------------------------------------------
# Test loop: compare equal-power vs PHASE-AWARE power
# -------------------------------------------------------------
@torch.no_grad()
def test_compare_power_phase_aware(policy_test,
                                   test_size,
                                   test_dataset_path,
                                   n_antennas_test,
                                   n_users,
                                   pinch_positions,
                                   parameters,
                                   sigma,            # noise std for user pos
                                   dev,
                                   K=100,            # MC samples per test instance
                                   P_tot=1.0,        # total power budget per user
                                   thr=0.5,
                                   alpha_proj=1.0,
                                   beta_attn=1.0):
    """
    For each test instance:
      - sample K noisy user positions, build graph batch, run policy
      - aggregate attentions, select active (mean prob > thr)
      - PHASE-AWARE power: allocate power by positive projection on dominant phasor,
        modulated by mean importance (exponents alpha_proj, beta_attn)
      - Compare to equal power across the same selected set
    Returns arrays of baseline (equal-power) and phase-aware power rates.
    """
    # Load data (true channels and reference info)
    user_positions_test, B_polar_test, B_true, _, _, _, a_opts_test = preprocess_data(
        dataset_path=test_dataset_path, total_samples=test_size, device=dev, use_noisy_B=False
    )

    original_pos_test = torch.from_numpy(user_positions_test).unsqueeze(1).to(torch.float32).to(dev)

    eq_rates = []     # equal-power rates
    opt_rates = []    # phase-aware power rates

    for i in range(test_size):
        # --- 1) Build K valid noisy user positions around the i-th true position ---
        noisy_user_pos = torch.zeros(K, 1, 3, device=dev)
        valid = 0
        while valid < K:
            cand = original_pos_test[i] + torch.randn_like(original_pos_test[i]) * sigma
            x, y, z = cand.squeeze(0).tolist()
            if -50 < x < 50 and -50 < y < 50 and 0 < z < 1:
                noisy_user_pos[valid] = cand
                valid += 1

        # --- 2) Compute B for the K noisy samples (physics model) ---
        B_noisy_np = beta_calc(
            parameters["N_PINCHES"],
            pinch_positions,
            noisy_user_pos.squeeze(1).detach().cpu().numpy(),
            parameters["LAMBDA"], parameters["LAMBDA_G"], parameters["PSI_0"],
            batch_size=K
        )
        B_noisy = torch.from_numpy(B_noisy_np).to(torch.complex64).to(dev)  # [K, N]

        # Reference channel for power allocation = mean over noisy samples
        B_ref_row = B_noisy.mean(dim=0)  # complex [N]

        # Build B_polar for the graph
        B_mag = torch.abs(B_noisy).unsqueeze(-1)
        B_phase = torch.angle(B_noisy).unsqueeze(-1)
        B_polar = torch.cat([B_mag, B_phase], dim=-1)                       # [K, N, 2]

        # --- 3) Build the graph batch for K samples and run the policy ---
        batch_graph = create_batch_graph(
            user_pos=noisy_user_pos, pinch_positions=pinch_positions, B=B_noisy, B_polar=B_polar, dev=dev
        )
        prob_K, imp_K = policy_test(batch_graph, parameters["N_PINCHES"], K)  # [K, U, N] each
        assert prob_K.size(1) == 1, "This script expects n_users == 1."

        # --- 4) Selection + PHASE-AWARE power split (aggregated across K) ---
        final_action_uN, p_alloc_uN, _, _ = select_and_split_power_phase_aware(
            prob_K=prob_K, imp_K=imp_K, B_ref_row=B_ref_row, P_tot=P_tot,
            thr=thr, alpha_proj=alpha_proj, beta_attn=beta_attn
        )  # shapes: [U,N], [U,N]

        # --- 5) Equal power over the same selected set (same P_tot) ---
        sel = final_action_uN.bool()  # [U, N]
        K_sel = sel.sum(dim=-1, keepdim=True).clamp_min(1).to(torch.float32)  # [U,1]
        p_eq_uN = sel.float() * (torch.as_tensor(P_tot, device=sel.device, dtype=p_alloc_uN.dtype) / K_sel)  # [U,N]

        # --- 6) Evaluate both on the TRUE channel for this instance ---
        B_true_row = B_true[i].to(dev).to(torch.complex64)  # [N]
        _, rate_eq  = snr_and_rate_with_power(B_true_row, p_eq_uN.squeeze(0))
        _, rate_opt = snr_and_rate_with_power(B_true_row, p_alloc_uN.squeeze(0))

        eq_rates.append(rate_eq.item())
        opt_rates.append(rate_opt.item())

    return np.array(eq_rates), np.array(opt_rates)


# -------------------------------------------------------------
# Main: load model, run comparison, and plot the graph
# -------------------------------------------------------------
if __name__ == "__main__":
    # Device & seeds
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)
    np.random.seed(42)

    # Setup
    n_users = 1
    n_antennas = 50
    test_size = 100        # <-- change as you like
    K = 100                # noisy samples per test instance
    sigma = 0           # user-position noise std
    P_tot = 1.0            # total power budget per user (same for both methods)
    thr = 0.5              # selection threshold on mean probability

    # Phase-aware hyperparameters (tune if you want)
    alpha_proj = 1.0       # weight of projection strength
    beta_attn  = 1.0       # weight of model importance (set 0.0 to ignore attention)

    # Parameters + geometry
    parameters = get_antenna_parameters(n_antennas, n_users)
    pinch_positions = generate_normalized_pinch_positions(parameters)

    # Model
    policy = Policy(in_chnl=1, hid_chnl=128, n_users=n_users,
                    key_size_embd=64, key_size_policy=64, val_size=64,
                    clipping=10, dev=dev).to(dev)

    # Load your trained weights
    model_path = "Model_Hybrid_loss_lambda2.pth"   # <-- set your checkpoint path
    chkpt = torch.load(model_path, map_location=dev, weights_only=False)
    policy.load_state_dict(chkpt["model_state_dict"])
    policy.eval()

    # Dataset
    test_dataset_path = f"data/test/augmented_dataset_{test_size}samples_{n_antennas}ant_SQUARE100_waveguide50_NEWSNR.json"

    # ---- Run comparison ----
    eq_rates, opt_rates = test_compare_power_phase_aware(
        policy_test=policy,
        test_size=test_size,
        test_dataset_path=test_dataset_path,
        n_antennas_test=n_antennas,
        n_users=n_users,
        pinch_positions=pinch_positions,
        parameters=parameters,
        sigma=sigma,
        dev=dev,
        K=K,
        P_tot=P_tot,
        thr=thr,
        alpha_proj=alpha_proj,
        beta_attn=beta_attn
    )

    # ---- Report & plot ----
    mean_eq  = eq_rates.mean()
    mean_opt = opt_rates.mean()
    rel_gain = (mean_opt - mean_eq) / max(mean_eq, 1e-9) * 100.0

    print(f"\nAverage Rate (equal power):        {mean_eq:.4f} bits/s/Hz")
    print(f"Average Rate (phase-aware power):  {mean_opt:.4f} bits/s/Hz")
    print(f"Relative gain:                     {rel_gain:.2f}%")

    # Sort instances by equal-power rate for a clean line plot
    order = np.argsort(eq_rates)
    x = np.arange(eq_rates.shape[0])
    plt.figure(figsize=(9,5))
    plt.plot(x, eq_rates[order],  label="Equal power (selected set)", linewidth=2)
    plt.plot(x, opt_rates[order], label="Phase-aware power", linewidth=2)
    plt.xlabel("Test instance (sorted by equal-power rate)")
    plt.ylabel("Rate  [bits/s/Hz]")
    plt.title(f"Rate comparison: equal vs phase-aware power (K={K}, sigma={sigma})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
