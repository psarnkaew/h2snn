import numpy as np, importlib.util, io, contextlib
spec = importlib.util.spec_from_file_location("cf", "check_face.py")
with contextlib.redirect_stdout(io.StringIO()):
    cf = importlib.util.module_from_spec(spec); spec.loader.exec_module(cf)
d, idx, edges, rng = cf.d, cf.idx, cf.edges, np.random.default_rng(1)
A = cf.A
D0, D1, P = cf.build(A)
# rebuild maps the same way build() does, to get T_v and per-edge target maps
F = dict(cf.F)
T = {"P1": A}
for i, e in enumerate([0, 1]):
    u, v = ["P1", "A7", "P9"][i], ["P1", "A7", "P9"][i + 1]
    T[v] = T[u] @ np.linalg.inv(F[(u, e)]) @ F[(v, e)]
F[("P1", 3)] = np.eye(d); F[("P9", 3)] = np.linalg.inv(T["P1"]) @ T["P9"]
x = rng.normal(size=(len(cf.nodes), d))
# DNSD 'adjacency' node->edge: edge state = target-side transported node state only
s_adj = np.concatenate([F[(v, k)] @ x[idx[v]] for k, (u, v) in enumerate(edges)])
face_adj = D1 @ s_adj
hub = sum(T[n] @ x[idx[n]] for n in ["P1", "A7", "P9"])  # (l+1) * z_f of v2 = MAGNN encoder sum
print("DNSD-style node->edge, then edge->face  vs  v2 hub sum:", np.abs(face_adj - hub).max())
# coboundary node->edge (difference): face sees nothing from nodes
s_cob = D0 @ x.reshape(-1)
print("coboundary node->edge, then edge->face:", np.abs(D1 @ s_cob).max())
