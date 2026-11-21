import numpy as np
import torch as t

# talvez n precise dessas duas
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

import gbfs_globals as GG
from kshell_2 import kshell_2
from sequential_strategies import apply_kshell_sequential

relevances_global = None
m_redundancy_global = None
m_synergy_global = None
data_tensor_global = None
y_tensor_global = None
d_total_global = None

def calc_entropy(x, bins=20):
    hist = t.histogram(x, bins=bins)[0]
    px = hist / hist.sum()
    mask = px > 0
    return -t.sum(px[mask] * t.log(px[mask])).item()

def calc_mutual_information(x, y, bins=20):
    # empilha duas features e calcula o diagrama 2d
    hist, _ = t.histogramdd(t.stack([x, y], dim=1), bins=bins)
    pxy = hist / hist.sum() # p(x,y) = probabilidade conjunta
    px = pxy.sum(dim=1)
    py = pxy.sum(dim=0)
    px_py = px.unsqueeze(1) * py.unsqueeze(0) # p(x) * p(y)
    mask = pxy > 0 # para evitar divisao por 0 ou log 0
    return t.sum(pxy[mask] * t.log(pxy[mask] / px_py[mask])).item()

def calc_joint_mutual_information(xi, xj, y, bins=10):
    hist, _ = t.histogramdd(t.stack([xi, xj, y], dim=1), bins=bins)
    pxx_y = hist / hist.sum() # p(xi, xj, j)
    pxx = pxx_y.sum(dim=2) # p(xi, xj)
    py = pxx_y.sum(dim=(0,1)) # p(y)
    # trocando linha com coluna
    pxx_py = pxx.unsqueeze(2) * py.unsqueeze(0).unsqueeze(0) #criando matriz de probabilidade conjunta, colocando mais dimensões
    mask = pxx_y > 0
    return t.sum(pxx_y[mask] * t.log(pxx_y[mask] / pxx_py[mask])).item()

def inicialize_informational_metrics():
    global relevances_global, m_redundancy_global, m_synergy_global
    global data_tensor_global, y_tensor_global, d_total_global

    data_tensor_global = t.tensor(GG.trData, dtype=t.float32) # mudando os dados do grafo para pytorch
    y_tensor_global = t.tensor(GG.trLabel.ravel(), dtype=t.float32)
    d_total_global = GG.featNum

    y_entropy = calc_entropy(y_tensor_global)
    
    relevances = np.zeros(d_total_global)
    entropies = np.zeros(d_total_global)
    
    for i in range(d_total_global):
        feature_i = data_tensor_global[:, i]
        h_x = calc_entropy(feature_i)
        mi = calc_mutual_information(feature_i, y_tensor_global)
        relevances[i] = mi / np.sqrt(h_x * y_entropy) if (h_x * y_entropy) > 0 else 0
        entropies[i] = h_x

    redundancy_matrix = np.zeros((d_total_global, d_total_global))
    synergy_matrix = np.zeros((d_total_global, d_total_global))

    for i in range(d_total_global):
        for j in range(i + 1, d_total_global):
            feature_i = data_tensor_global[:, i]
            feature_j = data_tensor_global[:, j]
            
            # Redundância: I(Xi; Xj)
            mi_ij = calc_mutual_information(feature_i, feature_j)
            redundancy_matrix[i, j] = mi_ij
            redundancy_matrix[j, i] = mi_ij
            
            # Sinergia
            joint_mi = calc_joint_mutual_information(feature_i, feature_j, y_tensor_global)
            mi_u_v = relevances[i] * np.sqrt(entropies[i] * y_entropy)
            gain = max(0, joint_mi - mi_u_v)
            synergy_val = mi_ij * gain
            
            synergy_matrix[i, j] = synergy_val
            synergy_matrix[j, i] = synergy_val

    relevances_global = relevances
    m_redundancy_global = redundancy_matrix
    m_synergy_global = synergy_matrix

def evaluate_subset(S_indices):
    if len(S_indices) == 0:
        return 0.0, float('inf'), 0.0, float('inf')

    # relevancia 
    f1_relevance = np.mean(relevances_global[S_indices])
    
    if len(S_indices) == 1: 
        return f1_relevance, 0.0, 0.0, 1.0 / d_total_global
    
    triu_idx = np.triu_indices(len(S_indices), k=1)
    
    # redundancia 
    sub_red = m_redundancy_global[np.ix_(S_indices, S_indices)]
    f2_redundancy = np.mean(sub_red[triu_idx])

    # sinergia 
    sub_syn = m_synergy_global[np.ix_(S_indices, S_indices)]
    f3_synergy = np.mean(sub_syn[triu_idx])

    # tamanho 
    f4_sparsity = len(S_indices) / d_total_global

    return f1_relevance, f2_redundancy, f3_synergy, f4_sparsity 

def calculate_f5(target_subset_idx, B=10):
    target_set = set(target_subset_idx)
    k = len(target_subset_idx)
    if k == 0: return 0.0
    jaccard_scores = []
    n_samples = data_tensor_global.shape[0]
    for b in range(B):
        idx_b = np.random.choice(n_samples, size=n_samples, replace=True)
        data_b = data_tensor_global[idx_b]
        y_b = y_tensor_global[idx_b]
        rels_b = [calc_mutual_information(data_b[:, i], y_b) for i in range (d_total_global)]
        selected_b = []
        unselected = list(range(d_total_global))
        first = max(unselected, key=lambda i: rels_b[i])
        selected_b.append(first)
        unselected.remove(first)
        
        while len(selected_b) < k:
            best_score = -float('inf')
            best_feat = -1
            for i in unselected:
                rel = rels_b[i]
                red_sum = sum([calc_mutual_information(data_b[:, i], data_b[:, s]) for s in selected_b])
                mid_score = rel - (red_sum / len(selected_b)) # Relevância - Redundância
                if mid_score > best_score:
                    best_score = mid_score
                    best_feat = i
            selected_b.append(best_feat)
            unselected.remove(best_feat)
        set_b = set(selected_b)
        intersection = len(target_set.intersection(set_b))
        union = len(target_set.union(set_b))
        jaccard_scores.append(intersection / union)
        
    return np.mean(jaccard_scores)

def evaluate_objective_f_internal(featIdx):
    rel, red, syn, size = evaluate_subset(featIdx)
    stab = calculate_f5(featIdx, B=5) 
    return np.array([-rel, red, -syn, size, -stab], dtype=float)

def evaluate_objective_f(M, individual_adj):
    corrMatrix = np.asarray(individual_adj, dtype=float)

    if not np.any(corrMatrix):
        if GG.data is None:
            raise ValueError("evaluate_objective_f: global 'data' is not set.")
        n_features = GG.data.shape[1]

        f = np.array([np.inf, np.inf, np.inf, np.inf, np.inf], dtype=float)
        featIdx_full = np.ones(n_features, dtype=bool)
        return f, featIdx_full

    if GG.trData is None or GG.trLabel is None:
        raise ValueError("evaluate_objective_f: global 'trData' or 'trLabel' is not set.")

    featIdx_core = kshell_2(corrMatrix)
    featIdx_core = np.asarray(featIdx_core).astype(int).ravel()

    featIdx_ext = apply_kshell_sequential(featIdx_core)

    if not np.array_equal(featIdx_core, featIdx_ext):
        f_1 = evaluate_objective_f_internal(featIdx_core)
        f_2 = evaluate_objective_f_internal(featIdx_ext)

        # ESSE TRECHO IMPLEMENTA DO MESMO MODO Q O ARTIGO BASE IMPLEMENTOU,
        #   MAS COMO ESSE MODO NÃO USA A DOMINANCIA DE PARETO PARA DETERMINAR A
        #       MELHOR SOLUÇÃO, MUDEI
        # def score(fv):
        #     w_rel = 0.2; w_red = 0.2; w_syn = 0.2; w_size = 0.2; w_stab = 0.2 #peso de cada objetivo
        #     return (w_rel * fv[0]) + (w_red * fv[1]) + (w_syn * fv[2]) + (w_size * fv[3]) + (w_stab * fv[4])

        # if score(f_1) < score(f_2):
        #     featIdx = featIdx_core
        #     f = f_1
        # else:
        #     featIdx = featIdx_ext
        #     f = f_2

        # IMPLEMENTANDO BASEADO NA DOMINANCIA DE PARETO
        f1_dominates = np.all(f_1 <= f_2) and np.any(f_1 < f_2)
        f2_dominates = np.all(f_2 <= f_1) and np.any(f_2 < f_1)

        if f1_dominates:
            featIdx = featIdx_core # conjunto original é melhor
            f = f_1
        elif f2_dominates:
            featIdx = featIdx_ext # conjunto com novas features adicionadas é melhor
            f = f_2
        else: # nenhuma é estritatmente melhor q a outra
            if np.linalg.norm(f_1) < np.linalg.norm(f_2): # usa norma euclidiana para desempatar, a solu mais perto da origem ganha
                featIdx = featIdx_core
                f = f_1
            else:
                featIdx = featIdx_ext
                f = f_2
    else:
        featIdx = featIdx_core
        f = evaluate_objective_f_internal(featIdx)

    if GG.data is None:
        raise ValueError("evaluate_objective_f: global 'data' is not set.")
    n_features = GG.data.shape[1]
    featIdx_full = np.zeros(n_features, dtype=bool)
    featIdx_full[featIdx] = True

    if f.size != M:
        raise ValueError("Objective size != M in evaluate_objective_f")

    return f, featIdx_full