import numpy as np
import pandas as pd
import repitl.matrix_itl as itl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from matplotlib import cm 
from typing import List, Union, Tuple
import torch.nn.functional as F
import seaborn as sns
import matplotlib.colors as mcolors
def aggregate_and_average_spectrum(all_runs_data):
    df_list = []
    for run_id, data in enumerate(all_runs_data):
        df = pd.DataFrame(data)
        df['run_id'] = run_id
        df_list.append(df)
        
    full_df = pd.concat(df_list, ignore_index=True)
    
    avg_df = full_df.groupby(['layer_index', 'svd_n'])['eigenvalue'].mean().reset_index()

    avg_df.rename(columns={'eigenvalue': 'avg_eigenvalue'}, inplace=True)

    return avg_df

def compute_spectrum_data(hs_list, center=True):
    layer_index_list = []
    svd_n_list = []
    eigenvalue_list = []
    
    eps = 1e-12

    for layer_idx, hs in enumerate(hs_list):
        if hs.dim() == 3:
            B, T, D = hs.shape
            Z = hs.reshape(B * T, D)
        elif hs.dim() == 2:
            Z = hs
        else:
            raise ValueError(f"Unsupported hidden_states dim: {hs.dim()} (need 2D or 3D)")

        Z = Z.to(torch.float64)
        if center:
            Z = Z - Z.mean(dim=0, keepdim=True)

        N, D = Z.shape

        if N >= D:
            G = Z.T @ Z  # [D, D]
            d = D       
        else:
            G = Z @ Z.T  # [N, N]
            d = N        

        G = torch.clamp(G, min=0)
            
        evals = torch.linalg.eigvalsh(G)
        evals = torch.clamp(evals, min=0)

        trace = torch.sum(evals)
        evals=evals/trace

        sorted_evals = torch.flip(evals, dims=[0])
            
        valid_evals = sorted_evals[sorted_evals > eps]
        num_valid_evals = valid_evals.numel()
            
        svd_n = np.arange(1, num_valid_evals + 1)
            
        current_layer_indices = np.full(num_valid_evals, layer_idx)

        layer_index_list.extend(current_layer_indices.tolist())
        svd_n_list.extend(svd_n.tolist())
        eigenvalue_list.extend(valid_evals.cpu().numpy().tolist())

    return {
        'layer_index': layer_index_list,
        'svd_n': svd_n_list,
        'eigenvalue': eigenvalue_list,
    }

def calculate_average_metrics(whole_spectrum_list):
    all_runs_dfs = []
    for run_id, metrics_dict in enumerate(whole_spectrum_list):
        df_run = pd.DataFrame(metrics_dict)

        df_run['layer_index'] = df_run.index 

        df_run['run_id'] = run_id
        
        all_runs_dfs.append(df_run)

    full_df = pd.concat(all_runs_dfs, ignore_index=True)
    
    full_df['layer_index'] = full_df['layer_index'].astype(int)

    metric_columns = [col for col in full_df.columns if col not in ['layer_index', 'run_id']]

    average_metrics_df = full_df.groupby('layer_index')[metric_columns].mean()
    
    return average_metrics_df
def compute_truncated_entropy(
    hs_list,
    k_method="topk",      # 'kaiser' | 'energy' | 'topk' | 'rho_kaiser'
    energy_ratio=0.95,      
    k_fixed=4,           
    log_base='e',           # 'e' or '2'
    center=True,
    eps=1e-12,
    tol=0.0,               
):
    H1_list, H1norm_list, H2_list = [], [], []
    eRank_list= []
    R_e_list, d_list = [], []
    k_list, thr_list = [], []

    ln2 = np.log(2.0)

    for hs in hs_list:
        try:
            if hs.dim() == 3:
                B, T, D = hs.shape
                Z = hs.reshape(B*T, D)
            elif hs.dim() == 2:
                Z = hs
            else:
                raise ValueError(f"Unsupported hidden_states dim: {hs.dim()}")

            Z = Z.to(torch.float64)
            if center:
                Z = Z - Z.mean(dim=0, keepdim=True)

            N, D = Z.shape
            if N >= D:
                G = Z.T @ Z   # [D, D]
                d = D
            else:
                G = Z @ Z.T   # [N, N]
                d = N

            G = torch.clamp(G, min=0)
            evals = torch.linalg.eigvalsh(G)
            evals = torch.clamp(evals, min=0)
            lam = torch.flip(evals, dims=[0])
            tr = lam.sum()

            if (not torch.isfinite(tr)) or (tr <= eps):
                raise RuntimeError("trace too small or non-finite")

            thr = None
            if k_method == "kaiser":
                thr = (tr / d) * (1.0 + tol)
                k = int((lam > thr).sum().item())
            elif k_method == "rho_kaiser":
                lam_rho = lam / tr
                thr = (1.0 / d) * (1.0 + tol)
                k = int((lam_rho > thr).sum().item())
            elif k_method == "energy":
                csum = torch.cumsum(lam, dim=0)
                target = energy_ratio * tr
                idx = torch.nonzero(csum >= target, as_tuple=False)
                k = int(idx[0, 0].item()+1) if idx.numel() > 0 else d
            elif k_method == "topk":
                if (k_fixed is None) or (int(k_fixed) <= 0):
                    raise ValueError("k_fixed must be a positive int for 'topk'")
                k = int(min(max(1, k_fixed), d))
            else:
                raise ValueError("k_method must be 'kaiser' | 'rho_kaiser' | 'energy' | 'topk'")

            if k <= 0: k = 1
            if k > d:  k = d

            lam_k = lam[:k]
            s_k = lam_k.sum()
            if s_k <= eps:
                raise RuntimeError("sum of top-k eigenvalues too small")

            p = lam_k / s_k                   
            p = torch.clamp(p, min=eps)

            # H1 (nats)
            H1_nats = float((-p * torch.log(p)).sum().item())
            # H2 (nats) = -log sum p^2
            H2_nats = float((-torch.log((p*p).sum())).item())

            if log_base == 'e':
                H1 = H1_nats
                H2 = H2_nats
                eRank = float(np.exp(H1))
                logk  = np.log(k) if k > 1 else eps
            elif log_base == '2':
                H1 = H1_nats / ln2
                H2 = H2_nats / ln2
                eRank = float(2.0 ** H1)
                logk  = np.log2(k) if k > 1 else eps
            else:
                raise ValueError("log_base must be 'e' or '2'")

            H1_norm = float(H1 / (logk + eps))

            R_e  = float(1.0 - eRank / k)

            H1_list.append(H1)
            H2_list.append(H2)
            H1norm_list.append(H1_norm)
            eRank_list.append(eRank)
            R_e_list.append(R_e)
            d_list.append(int(d))
            k_list.append(int(k))
            thr_list.append(float(thr) if thr is not None else np.nan)

        except Exception:
            H1_list.append(float('nan'))
            H2_list.append(float('nan'))
            H1norm_list.append(float('nan'))
            eRank_list.append(float('nan'))
            R_e_list.append(float('nan'))
            d_list.append(int(d) if 'd' in locals() else 0)
            k_list.append(0)
            thr_list.append(float('nan'))

    return {
        'H1_trunc_raw': H1_list,
        'H1_trunc_maxEntropy': H1norm_list,  # = H1 / log(k)
        'H2_trunc_raw': H2_list,
        'eRank_trunc': eRank_list,
        'redundancy_erank_trunc': R_e_list,   # 1 - eRank / k
        'ambient_dim': d_list,
        'k_used': k_list,
        'k_threshold': thr_list,
    }
        
def plot_3d_average_spectrum(avg_df, save_path=None):
    plt.rcParams['figure.dpi'] = 150 
    plt.rcParams['savefig.dpi'] = 300
    fig = plt.figure(figsize=(8, 6), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    grid_color = (0.8, 0.8, 0.8, 1) 
    ax.xaxis._axinfo["grid"]['color'] = grid_color
    ax.yaxis._axinfo["grid"]['color'] = grid_color
    ax.zaxis._axinfo["grid"]['color'] = grid_color
    ax.xaxis._axinfo["grid"]['linestyle'] = "--"
    ax.yaxis._axinfo["grid"]['linestyle'] = "--"
    ax.zaxis._axinfo["grid"]['linestyle'] = "--"
    
    colors = ["#4E7989", "#8FA876", "#F0C458"]

    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_paper_style", colors)
    
    colormap = custom_cmap

    unique_layers = sorted(avg_df['layer_index'].unique())
    norm = plt.Normalize(vmin=min(unique_layers), vmax=max(unique_layers))

    for layer, group in avg_df.groupby('layer_index'):
        color = colormap(norm(layer))
        
        ax.plot(group['layer_index']-1,
                group['svd_n'], 
                group['avg_eigenvalue'], 
                label=f'Layer {int(layer)-1}', 
                linewidth=2, 
                alpha=0.9,    
                color=color) 

    text_color = '#333333'
    
    ax.set_xlabel('Layer Index', fontsize=12, labelpad=4, fontweight='bold', color=text_color)
    ax.set_ylabel('SVD n (Rank Index)', fontsize=12, labelpad=4, fontweight='bold', color=text_color)
    ax.set_zlabel('Eigenvalue', fontsize=12, labelpad=4, fontweight='bold', color=text_color)

    ax.tick_params(axis='x', colors=text_color)
    ax.tick_params(axis='y', colors=text_color)
    ax.tick_params(axis='z', colors=text_color)

    sparse_layers = [
        layer for layer in unique_layers 
        if int(layer) % 4 == 0
    ]   
    ax.set_xticks(sparse_layers)

    min_layer_val = avg_df['layer_index'].min()
    max_layer_val = avg_df['layer_index'].max()
    ax.set_xlim(max_layer_val, min_layer_val)

    ax.set_box_aspect((1.4, 1.6, 1))
    ax.view_init(elev=20, azim=30) 

    plt.savefig(save_path) if save_path else None
