import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

def _neighbors_from_adj(A, tol=0.0):
    """
    Cria uma lista de vértices adjacentes a partir da matriz de adjacência A.
    neighbors[i] = np.array com os índices j para os quais A[i, j] > tol
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    neighbors = []
    for i in range(n):
        neigh_i = np.where(A[i] > tol)[0]
        neighbors.append(neigh_i.astype(int))
    return neighbors


def init_graph_local_threshold( #constroi a topologia inicial do grafo
    Z,
    k_min=1,
    quantile=0.8,
    exclude_self=True,
    symmetrize=True,
):
    """
    Constrói o grafo com um limiar específico para cada vértice:
      - Para cada característica i, seleciona as arestas com sim >= quantil da linha i.
      - Se o número de arestas for < k_min, adiciona as k_min maiores similaridades.

    Parameters
    ----------
    Z : np.ndarray, shape (n_features, n_features)
        Matriz de similaridade (por exemplo, GG.Zout).
    k_min : int
        Número mínimo de vizinhos de cada vértice.
    quantile : float in (0,1)
        Limiar de quantil para cada linha; por exemplo, 0.8 seleciona os 20% maiores valores de similaridade.
    exclude_self : bool
        Se True, remove a aresta (i, i).
    symmetrize : bool
        Se True, define A = max(A, A^T) para tornar o grafo não direcionado.

    Returns
    -------
    A : np.ndarray, shape (n_features, n_features)
        Matriz de adjacência (0 = sem aresta).
    neighbors : list[np.ndarray]
        neighbors[i] = lista dos vértices adjacentes a i (índices das características).
    """
    Z = np.asarray(Z, dtype=float)
    n = Z.shape[0]
    A = np.zeros_like(Z, dtype=float)

    for i in range(n):
        row = Z[i].copy()
        if exclude_self:
            row[i] = 0.0

        positive = row[row > 0]
        if positive.size == 0:
            continue

        tau_i = np.quantile(positive, quantile)
        neigh = np.where(row >= tau_i)[0]

        if exclude_self:
            neigh = neigh[neigh != i]

        # Garante pelo menos k_min vizinhos
        if neigh.size < k_min:
            order = np.argsort(-row)  # ordem decrescente
            if exclude_self:
                order = order[order != i]
            neigh = order[:k_min]

        for j in neigh:
            if i == j:
                continue
            A[i, j] = Z[i, j]

    if symmetrize:
        A = np.maximum(A, A.T)

    neighbors = _neighbors_from_adj(A)
    return A, neighbors


def init_graph_fisher_redundancy_degree(
    Z,
    fisher_scores,
    k_min=1,
    k_max=20,
    exclude_self=True,
    symmetrize=True,
):
    """
    O número de vizinhos deg_i de cada vértice i depende de:
      - escore de Fisher f_i (quanto maior, mais arestas)
      - redundância r_i (quanto maior, menos arestas)

    deg_i = k_min + (k_max - k_min) * [0.5*f_norm_i + 0.5*(1 - r_norm_i)]

    Parameters
    ----------
    Z : np.ndarray, (n, n)
        Similarity matrix.
    fisher_scores : np.ndarray, (n,)
        Escore de Fisher para cada característica (antes da normalização).
    k_min, k_max : int
        Número mínimo/máximo de arestas por vértice.
    exclude_self, symmetrize : bool

    Returns
    -------
    A : np.ndarray, (n, n)
        Adjacency matrix.
    degs : np.ndarray, (n,)
        Número de vizinhos selecionados para cada vértice (antes da simetrização).
    neighbors : list[np.ndarray]
        neighbors[i] conforme A (após a simetrização, se symmetrize=True).
    """
    Z = np.asarray(Z, dtype=float)
    fisher_scores = np.asarray(fisher_scores, dtype=float)
    n = Z.shape[0]
    assert fisher_scores.shape[0] == n

    # redundância r_i = similaridade média com os outros vértices
    R = np.abs(Z.copy())
    if exclude_self:
        np.fill_diagonal(R, 0.0)
    r_i = R.mean(axis=1)

    # Normaliza o escore de Fisher e a redundância para [0, 1]
    f_min, f_max = fisher_scores.min(), fisher_scores.max()
    if f_max - f_min < 1e-12:
        f_norm = np.full_like(fisher_scores, 0.5, dtype=float)
    else:
        f_norm = (fisher_scores - f_min) / (f_max - f_min)

    r_min, r_max = r_i.min(), r_i.max()
    if r_max - r_min < 1e-12:
        r_norm = np.full_like(r_i, 0.5, dtype=float)
    else:
        r_norm = (r_i - r_min) / (r_max - r_min)

    score = 0.5 * f_norm + 0.5 * (1.0 - r_norm)

    k_max = min(k_max, n - 1)
    degs_float = k_min + (k_max - k_min) * score
    degs = np.clip(np.round(degs_float).astype(int), k_min, k_max)

    A = np.zeros_like(Z, dtype=float)
    for i in range(n):
        row = Z[i].copy()
        if exclude_self:
            row[i] = 0.0

        order = np.argsort(-row)  # ordem decrescente
        if exclude_self:
            order = order[order != i]

        ki = int(degs[i])
        if ki <= 0 or order.size == 0:
            continue

        ki = min(ki, order.size)
        neigh = order[:ki]
        for j in neigh:
            A[i, j] = Z[i, j]

    if symmetrize:
        A = np.maximum(A, A.T)

    neighbors = _neighbors_from_adj(A)
    return A, degs, neighbors


def init_graph_mst_plus_local(
    Z,
    extra_k=2,
    exclude_self=True,
    symmetrize=True,
):
    """
    Constrói o grafo em duas etapas:
      1) Estrutura principal MST (com distância = 1 - similaridade).
      2) Adiciona a cada vértice as extra_k arestas locais mais fortes
         (sem repetir arestas da MST).

    Parameters
    ----------
    Z : np.ndarray, (n, n)
        Matriz de similaridade.
    extra_k : int
        Número de vizinhos locais adicionados a cada vértice além da MST.
    exclude_self, symmetrize : bool

    Returns
    -------
    A : np.ndarray, (n, n)
        Matriz de adjacência após adicionar a MST e as arestas locais.
    neighbors : list[np.ndarray]
        neighbors[i] conforme A (após a simetrização, se symmetrize=True).
    """
    Z = np.asarray(Z, dtype=float)
    n = Z.shape[0]

    # 1) MST com distância = 1 - similaridade
    D = 1.0 - Z
    D[D < 0.0] = 0.0

    mst_sparse = minimum_spanning_tree(D)
    mst = mst_sparse.toarray()
    # MST simétrica (não direcionada)
    mst = np.maximum(mst, mst.T)

    A = np.zeros_like(Z, dtype=float)

    # Adiciona as arestas da MST com pesos de similaridade
    i_idx, j_idx = np.where(mst > 0)
    for i, j in zip(i_idx, j_idx):
        if i == j:
            continue
        A[i, j] = Z[i, j]
        A[j, i] = Z[j, i]

    # 2) Adiciona os extra_k vizinhos locais mais fortes a cada vértice
    for i in range(n):
        row = Z[i].copy()
        if exclude_self:
            row[i] = 0.0

        # Remove os nós já conectados pela MST
        already = np.where(A[i] > 0)[0]
        if already.size > 0:
            row[already] = 0.0

        if not np.any(row > 0):
            continue

        order = np.argsort(-row)
        if exclude_self:
            order = order[order != i]

        ki = min(extra_k, order.size)
        neigh_extra = order[:ki]

        for j in neigh_extra:
            if i == j:
                continue
            A[i, j] = Z[i, j]
            A[j, i] = Z[j, i]

    if symmetrize:
        A = np.maximum(A, A.T)

    neighbors = _neighbors_from_adj(A)
    return A, neighbors


def init_graph_probabilistic(
    Z,
    fisher_scores,
    k_min=1,
    k_max=10,
    beta=2.0,
    exclude_self=True,
    symmetrize=True,
    random_state=None,
):
    """
    Para cada vértice i:
      - O grau deg_i = k_min + (k_max - k_min) * f_norm_i (Fisher maior -> mais arestas).
      - Seleciona deg_i vizinhos por amostragem sem reposição, com probabilidade:
            p_ij ∝ (sim_ij ** beta) * g(f_i, f_j)
        em que g(f_i, f_j) ~ 0.5*(f_norm_i + f_norm_j).
      - Se todos os pesos forem zero, recorre às k maiores similaridades.

    Parameters
    ----------
    Z : np.ndarray, (n, n)
        Similarity matrix.
    fisher_scores : np.ndarray, (n,)
        Fisher scores.
    k_min, k_max : int
        Grau mínimo/máximo de cada vértice.
    beta : float
        Grau de concentração da distribuição baseada na similaridade (beta maior -> maior preferência por arestas fortes).
    exclude_self, symmetrize : bool
    random_state : int or None
        Semente para reprodutibilidade.

    Returns
    -------
    A : np.ndarray, (n, n)
        Adjacency matrix.
    neighbors : list[np.ndarray]
        neighbors[i] conforme A (após a simetrização, se symmetrize=True).
    """
    Z = np.asarray(Z, dtype=float)
    fisher_scores = np.asarray(fisher_scores, dtype=float)
    n = Z.shape[0]
    assert fisher_scores.shape[0] == n

    rng = np.random.default_rng(random_state)

    # normalize Fisher
    f_min, f_max = fisher_scores.min(), fisher_scores.max()
    if f_max - f_min < 1e-12:
        f_norm = np.full_like(fisher_scores, 0.5, dtype=float)
    else:
        f_norm = (fisher_scores - f_min) / (f_max - f_min)

    k_max = min(k_max, n - 1)
    A = np.zeros_like(Z, dtype=float)

    indices_all = np.arange(n)

    for i in range(n):
        row = Z[i].copy()
        if exclude_self:
            row[i] = 0.0

        # Grau desejado para o nó i
        k_i_float = k_min + (k_max - k_min) * f_norm[i]
        k_i = int(round(k_i_float))
        k_i = max(k_min, min(k_i, n - 1))

        if k_i <= 0:
            continue

        vals = np.maximum(row, 0.0)

        # Se todos os pesos forem zero, recorre aos k maiores valores
        if not np.any(vals > 0):
            order = np.argsort(-row)
            if exclude_self:
                order = order[order != i]
            neigh = order[:k_i]
        else:
            # weight w_j = (sim_ij^beta) * g(f_i, f_j)
            g_ij = 0.5 * (f_norm[i] + f_norm)  # vector over j
            w = (vals ** beta) * g_ij

            if exclude_self:
                mask_valid = indices_all != i
                cand_idx = indices_all[mask_valid]
                w = w[mask_valid]
            else:
                cand_idx = indices_all

            if not np.any(w > 0):
                # fallback top-k theo similarity
                row2 = row.copy()
                if exclude_self:
                    row2[i] = 0.0
                order = np.argsort(-row2)
                if exclude_self:
                    order = order[order != i]
                neigh = order[:k_i]
            else:
                probs = w / w.sum()
                k_eff = min(k_i, cand_idx.size)
                neigh = rng.choice(cand_idx, size=k_eff, replace=False, p=probs)

        for j in neigh:
            if i == j:
                continue
            A[i, j] = Z[i, j]

    if symmetrize:
        A = np.maximum(A, A.T)

    neighbors = _neighbors_from_adj(A)
    return A, neighbors
