import numpy as np

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation

from scipy.spatial.distance import squareform

from evaluate_objective_f import evaluate_objective_f
from evaluate_objective_f import inicialize_informational_metrics
import gbfs_globals as GG

print("NEUWTRI")
def _get_pareto_indices(objs, minimize=True):
    """
    objs : np.ndarray, shape (N, M)  (N soluções, M objetivos)
    minimize : bool
        Se True, assume que todos os objetivos seguem a regra "quanto menor, melhor".

    Retorna:
        idx_front : np.ndarray com os índices das soluções não dominadas (fronteira de Pareto).
    """
    objs = np.asarray(objs, dtype=float)
    N = objs.shape[0]

    if not minimize:
        # Se algum objetivo for de maximização, inverta o sinal antes da chamada
        # ou trate esse caso separadamente aqui.
        pass

    is_nd = np.ones(N, dtype=bool)  # non-dominated flags

    for i in range(N):
        if not is_nd[i]:
            continue
        # solução i
        oi = objs[i]

        # a solução j é dominada por i se:
        #   - oi <= oj em todas as dimensões
        #   - oi <  oj em pelo menos uma dimensão
        # → aqui, buscamos as soluções (j) dominadas por i
        dominated = np.all(objs >= oi, axis=1) & np.any(objs > oi, axis=1)
        # mas NÃO excluímos a própria solução i
        dominated[i] = False
        is_nd[dominated] = False

    idx_front = np.where(is_nd)[0]
    return idx_front

class GBFS_featureSelection(Problem):
    def __init__(self, n_features, v_f, templateAdj, n_obj=2):
        self.n_features = n_features
        self.v_f = v_f

        if templateAdj.ndim == 1 and len(templateAdj) == (n_features * (n_features - 1)) // 2:
            self.templateAdj = squareform(templateAdj)
        else:
            # Caso venha normal (81 itens)
            self.templateAdj = templateAdj.reshape((n_features, n_features))

        #self.templateAdj = np.asarray(templateAdj, dtype=float).reshape((n_features, n_features))
        t_tot_cromossomo = n_features + v_f
        super().__init__(n_var = t_tot_cromossomo, n_obj=n_obj, n_ieq_constr=0, xl=0, xu=1, vtype=bool)

    def _evaluate(self, x, out, *args, **kwargs):
        pop_size = x.shape[0]
        objs = np.zeros((pop_size, self.n_obj))
        for i in range(pop_size):
            chromosome = x[i]
            feat_mask = chromosome[:self.n_features]
            kNeigh_mask = chromosome[self.n_features : self.n_features + self.v_f]

            individual_adj = self.templateAdj.copy() #copia do grafo candidato
            inactive_nodes = np.where(feat_mask == 0)[0]
            individual_adj[inactive_nodes, :] = 0
            individual_adj[:, inactive_nodes] = 0

            f_val, _ = evaluate_objective_f(self.n_obj, individual_adj)
            objs[i, :] = f_val
        out["F"] = objs

def newtry_ms(inputAdj, pop=20, times=40, run_dir=None):
    """
    Parameters
    ----------
    inputAdj : array-like
    pop : int
        Population size
    times : int
        Number of generations

    Returns
    -------
    featidx_best : np.ndarray, shape (featNum,)
        Máscara binária da melhor solução segundo a função de aptidão escalar (como no código antigo).
    pareto_masks : np.ndarray, shape (n_solutions, featNum)
        Máscaras 0/1 das soluções na fronteira de Pareto segundo (obj1, obj2).
    pareto_objs  : np.ndarray, shape (n_solutions, 2)
        Valores (obj1, obj2) correspondentes a cada solução de Pareto.
    """
    if GG.featNum is None:
        raise ValueError("newtry_ms: GG.featNum hoặc GG.kNeigh chưa được set.")
   
    V_f = int(sum(len(nb) for nb in GG.neigh_list))
    GG.V_f = V_f
    templateAdj = np.asarray(inputAdj, dtype=float)
    n_obj = 5
    ref_dirs = get_reference_directions("energy", n_dim=n_obj, n_points=pop)

    algorithm = NSGA3(
        pop_size=pop,
        ref_dirs=ref_dirs,
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(),
        mutation=BitflipMutation(),
        eliminate_duplicates=True
    )
    problem = GBFS_featureSelection(
        n_features=GG.featNum, 
        v_f=V_f, 
        templateAdj=templateAdj, 
        n_obj=n_obj
    )
    res = minimize(
        problem,
        algorithm,
        ('n_gen', times),
        seed=42, 
        verbose=False
    )
    pop_X = res.pop.get("X")
    pop_F = res.pop.get("F")

    # Supõe-se o formato: [feat_part | obj1 | obj2 | ...]
    feat_part = pop_X[:, :GG.featNum]
    kNeigh_part = pop_X[:, GG.featNum : GG.featNum + V_f]
    obj_part = pop_F

    ideal_point = np.min(obj_part, axis=0) # utopia point, usa distancia ao invés de 2 pesos fixos
    max_point = np.max(obj_part, axis=0)
    range_point = np.maximum(max_point - ideal_point, 1e-6)
    normalized_objs = (obj_part - ideal_point) / range_point # normaliza objetivos para estabilidade n sobrepor o erro

    distances = np.linalg.norm(normalized_objs, axis=1)
    idx_best = np.argmin(distances) # a melhor solução é a com menor dist euclidiana até o ideal point
    
    featidx_best = feat_part[idx_best, :]
    kNeigh_best = kNeigh_part[idx_best, :]
    # print("Best solution:")
    # print("Feature mask:", featidx_best)
    # print("kNeigh mask:", kNeigh_best)

    # -----------------------------
    # 2) Obtém a fronteira de Pareto com base em (obj1, obj2, obj3, obj4, obj5)
     # -----------------------------
    
    idx_front = _get_pareto_indices(obj_part, minimize=True)
    pareto_masks = feat_part[idx_front, :]
    pareto_objs  = obj_part[idx_front, :]

    return featidx_best, pareto_masks, pareto_objs, kNeigh_best
