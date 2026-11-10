"""Check on the v1 construction (manuscript Eqs. transport/closure/delta1):
1) chain condition  delta1 @ delta0 = 0
2) signed coupling  delta0^T delta1^T = 0  (Lemma A': face info cannot reach nodes via delta0^T)
3) unsigned one-sided coupling P (v1 readout pooling, DNSD 'adjacency' idea) gives P delta1^T != 0,
   and which nodes receive face information (incl. the intermediate A7)
4) A_f enters the face term only through A_f^T A_f (orthogonal A_f is a no-op)
Graph: P1 -(writes)-> A7 ... instance f = P1 - A7 - P9 with closure P9 -> P1, plus extra edge A7-P4.
"""
import numpy as np
rng = np.random.default_rng(0)
d = 3
nodes = ["P1", "A7", "P9", "P4"]
idx = {n: i for i, n in enumerate(nodes)}
# oriented edges (u -> v); e0,e1 path edges, e2 extra edge, e3 closure (v_l -> v_0)
edges = [("P1", "A7"), ("A7", "P9"), ("A7", "P4"), ("P9", "P1")]
N, E = len(nodes), len(edges)

def conf():  # scaled orthogonal map (manuscript Eq. conformal-map)
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return rng.uniform(0.5, 2.0) * q

F = {}  # F[(node, edge_id)]
for k, (u, v) in enumerate(edges[:3]):
    F[(u, k)] = conf(); F[(v, k)] = conf()

def build(A_f):
    # transport along path P1 -> A7 -> P9
    path_nodes = ["P1", "A7", "P9"]; path_edges = [0, 1]
    T = {"P1": A_f}
    Fef = {}
    for i, e in enumerate(path_edges):
        u, v = path_nodes[i], path_nodes[i + 1]
        Fu_inv = np.linalg.inv(F[(u, e)])
        Fef[e] = T[u] @ Fu_inv
        T[v] = T[u] @ Fu_inv @ F[(v, e)]
    # closure edge e3 = (P9 -> P1): F_{v0<=cl}=I, F_{vl<=cl}=T_v0^{-1}T_vl, F_{cl<=f}=A_f
    Fl = dict(F)
    Fl[("P1", 3)] = np.eye(d)
    Fl[("P9", 3)] = np.linalg.inv(T["P1"]) @ T["P9"]
    Fef[3] = A_f
    # delta0 : C0 -> C1, (delta0 x)_e = F_{v<=e} x_v - F_{u<=e} x_u  (u -> v)
    D0 = np.zeros((E * d, N * d))
    for k, (u, v) in enumerate(edges):
        if k == 3:  # closure oriented v_l -> v_0 : u = P9, v = P1
            pass
        D0[k*d:(k+1)*d, idx[v]*d:(idx[v]+1)*d] += Fl[(v, k)]
        D0[k*d:(k+1)*d, idx[u]*d:(idx[u]+1)*d] -= Fl[(u, k)]
    # delta1 : C1 -> C2 (one face), boundary (e0, e1, e3) with + signs
    D1 = np.zeros((d, E * d))
    for e in [0, 1, 3]:
        D1[:, e*d:(e+1)*d] = Fef[e]
    # unsigned one-sided edge->node pooling (v1 Eq. pool-transport, applied per layer)
    P = np.zeros((N * d, E * d))
    for k, (u, v) in enumerate(edges):
        for z in (u, v):
            P[idx[z]*d:(idx[z]+1)*d, k*d:(k+1)*d] += Fl[(z, k)].T
    return D0, D1, P

A = rng.normal(size=(d, d))
D0, D1, P = build(A)
print("1) ||delta1 delta0||          =", np.abs(D1 @ D0).max())
print("2) ||delta0^T delta1^T||      =", np.abs(D0.T @ D1.T).max())
M = P @ D1.T  # face value -> node
for n in nodes:
    print(f"3) ||(P delta1^T) at {n}||  =", np.abs(M[idx[n]*d:(idx[n]+1)*d]).max().round(4))
# face term on a random edge cochain, then pooled to nodes
s = rng.normal(size=E * d)
s_im = D0 @ rng.normal(size=N * d)
print("   face term on s in im(delta0) =", np.abs(D1.T @ D1 @ s_im).max(), "(face sees only non-gradient edge signal)")
# 4) A_f only via A_f^T A_f
Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
L_A = build(A)[1].T @ build(A)[1]
L_QA = build(Q @ A)[1].T @ build(Q @ A)[1]
print("4) ||L1up(A) - L1up(QA)||     =", np.abs(L_A - L_QA).max(), "(A_f only matters via A_f^T A_f)")
