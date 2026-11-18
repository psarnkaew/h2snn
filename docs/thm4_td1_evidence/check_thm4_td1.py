"""Numerical check of spec v1.25 Thm 4 (class-separating sections with TD-1 maps).

Builds the H2S-NN v3-B energy (node-fact, fact-cell, node-cell terms; forced face maps) as a
dense matrix from its terms, on a small random heterogeneous graph with heterophilic labels,
with TD-1 maps
    F_src = I,                 F_tgt = R_r exp(U_r C_e U_r^T)      (rho(r) != r)
    F_src = exp(-A_e / 2),     F_tgt = exp(+A_e / 2), A_e = U_r C_e U_r^T, C antisymmetric  (rho(r) = r)
(orthogonal factors; the positive scales c_e enter as weights a_e = c_e^2 and do not change sections).
It checks the 1.25 construction, its corollary for given vectors, and two negative controls.
float64 throughout; CPU; seconds. Run: python -I check_thm4_td1.py > check_thm4_td1.out
Design session, 2026-11-18. Not part of src/.
"""
import json
import math
import random

import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
random.seed(0)

TOL = 1e-10


def orthonormal(d, m):
    q, _ = torch.linalg.qr(torch.randn(d, m))
    return q


def random_so(d):
    q, r = torch.linalg.qr(torch.randn(d, d))
    q = q * torch.sign(torch.diagonal(r))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def gen(m):
    """J with exp(phi J) e0 = cos(phi) e0 + sin(phi) e1."""
    j = torch.zeros(m, m)
    j[1, 0], j[0, 1] = 1.0, -1.0
    return j


def exp_lowrank(u, c):
    """exp(U C U^T) = I + U (exp(C) - I) U^T for U^T U = I."""
    d, m = u.shape
    return torch.eye(d) + u @ (torch.linalg.matrix_exp(c) - torch.eye(m)) @ u.T


# ---------------------------------------------------------------- graph
def build_graph(n_per_type=14, n_edges_per_rel=24, n_faces=40, seed=1):
    rng = random.Random(seed)
    types = [t for t in range(3) for _ in range(n_per_type)]
    nodes_of = {t: [v for v, tv in enumerate(types) if tv == t] for t in range(3)}
    # relation -> (src type, tgt type, shared_role)
    relations = {0: (0, 1, False), 1: (1, 2, False), 2: (0, 0, True), 3: (2, 0, False)}
    edges, seen = [], set()
    for r, (ts, tt, _) in relations.items():
        while sum(1 for e in edges if e[2] == r) < n_edges_per_rel:
            u, v = rng.choice(nodes_of[ts]), rng.choice(nodes_of[tt])
            key = (r, min(u, v), max(u, v)) if relations[r][2] else (r, u, v)
            if u == v or key in seen:
                continue
            seen.add(key)
            edges.append((u, v, r))
    # every node gets at least one fact (degree 0 would leave D^{-1/2} undefined)
    out_rel = {0: 0, 1: 1, 2: 3}
    covered = {x for u, v, _ in edges for x in (u, v)}
    for w in range(len(types)):
        if w in covered:
            continue
        r = out_rel[types[w]]
        tgt = rng.choice(nodes_of[relations[r][1]])
        edges.append((w, tgt, r))
        covered.update((w, tgt))
    adj = {v: [] for v in range(len(types))}
    for ei, (u, v, _) in enumerate(edges):
        adj[u].append((ei, v))
        adj[v].append((ei, u))
    faces = []
    while len(faces) < n_faces:
        length = rng.randint(2, 4)
        walk, facts = [rng.randrange(len(types))], []
        ok = True
        for _ in range(length):
            nbrs = [(ei, w) for ei, w in adj[walk[-1]] if w not in walk]
            if not nbrs:
                ok = False
                break
            ei, w = rng.choice(nbrs)
            facts.append(ei)
            walk.append(w)
        if ok:
            faces.append((walk, facts))
    return types, relations, edges, faces


# ---------------------------------------------------------------- maps
def make_maps(edges, relations, d, frames, coords):
    """coords(e) -> (U_r, C_e). Returns per edge (F_src, F_tgt) orthogonal factors."""
    maps = []
    for ei, (u, v, r) in enumerate(edges):
        uu, c = coords(ei)
        a = uu @ c @ uu.T
        if relations[r][2]:
            maps.append((torch.linalg.matrix_exp(-a / 2), torch.linalg.matrix_exp(a / 2)))
        else:
            maps.append((torch.eye(d), frames[r] @ exp_lowrank(uu, c)))
    return maps


def fo(maps, edges, ei, node):
    u, v, _ = edges[ei]
    return maps[ei][0] if node == u else maps[ei][1]


# ---------------------------------------------------------------- energy matrix
def build_L(n_nodes, edges, faces, maps, a, beta, d):
    nE, nF = len(edges), len(faces)
    nb = n_nodes + nE + nF
    L = torch.zeros(nb * d, nb * d)

    def blk(i):
        return slice(i * d, (i + 1) * d)

    def add(w, i, m1, j, m2):  # w * || m1 Y_i - m2 Y_j ||^2
        L[blk(i), blk(i)] += w * m1.T @ m1
        L[blk(j), blk(j)] += w * m2.T @ m2
        L[blk(i), blk(j)] -= w * m1.T @ m2
        L[blk(j), blk(i)] -= w * m2.T @ m1

    eye = torch.eye(d)
    for ei, (u, v, _) in enumerate(edges):
        for node in (u, v):
            add(a[ei], node, fo(maps, edges, ei, node), n_nodes + ei, eye)
    transports = []
    for fi, (walk, facts) in enumerate(faces):
        zf = n_nodes + nE + fi
        t = eye.clone()
        ts = [t]
        add(beta[fi], walk[0], t, zf, eye)
        for j, ei in enumerate(facts):
            prev, cur = walk[j], walk[j + 1]
            r_ef = t @ fo(maps, edges, ei, prev).T                     # R_{e_j,f} = T_{v_{j-1}} F_{v_{j-1}<e}^T
            add(beta[fi], n_nodes + ei, r_ef, zf, eye)
            q = fo(maps, edges, ei, prev).T @ fo(maps, edges, ei, cur)  # Q_j
            t = t @ q                                                  # T_{v_j}
            ts.append(t)
            add(beta[fi], cur, t, zf, eye)
        transports.append(ts)
    return L, transports


def section(n_nodes, edges, faces, maps, x_of, d):
    ys = [x_of(v) for v in range(n_nodes)]
    ys += [fo(maps, edges, ei, u) @ x_of(u) for ei, (u, v, _) in enumerate(edges)]
    ys += [x_of(walk[0]) for walk, _ in faces]
    return torch.cat(ys)


def degrees(L, nb, d):
    out, worst = [], 0.0
    for i in range(nb):
        b = L[i * d:(i + 1) * d, i * d:(i + 1) * d]
        s = torch.trace(b) / d
        worst = max(worst, float((b - s * torch.eye(d)).abs().max()))
        out.append(float(s))
    return torch.tensor(out), worst


def anchored_solve(L, deg, n_nodes, h0, tau, d):
    dm = deg.repeat_interleave(d)
    lhat = L / torch.sqrt(dm)[:, None] / torch.sqrt(dm)[None, :]
    p = torch.zeros(n_nodes * d, L.shape[0])
    p[:, :n_nodes * d] = torch.eye(n_nodes * d)
    m = p.T @ p + tau * lhat
    y = torch.linalg.solve(m, p.T @ h0.reshape(-1))
    return y[:n_nodes * d].reshape(n_nodes, d), float(torch.linalg.eigvalsh(m).min())


def run_case(name, graph, labels, d, m, coords, frames, a_vecs, tau=7.0, perturb=None):
    types, relations, edges, faces = graph
    n = len(types)
    maps = make_maps(edges, relations, d, frames, coords)
    if perturb is not None:
        maps = perturb(maps)
    a = [0.2 + random.random() for _ in edges]
    beta = [0.05 + random.random() for _ in faces]
    L, transports = build_L(n, edges, faces, maps, a, beta, d)
    x_of = lambda v: a_vecs[labels[v]]
    y = section(n, edges, faces, maps, x_of, d)
    energy = 0.5 * float(y @ L @ y)
    ly = float((L @ y).abs().max())
    nb = n + len(edges) + len(faces)
    deg, scalar_dev = degrees(L, nb, d)
    # degree formulas of the spec: D_v = sum a_e + sum beta_f, D_e = 2 a_e + sum beta_f, D_f = beta_f (2l + 1)
    dv = torch.zeros(n); de = torch.tensor([2 * x for x in a]); df = torch.zeros(len(faces))
    for ei, (u, v, _) in enumerate(edges):
        dv[u] += a[ei]; dv[v] += a[ei]
    for fi, (walk, facts) in enumerate(faces):
        for v in walk:
            dv[v] += beta[fi]
        for ei in facts:
            de[ei] += beta[fi]
        df[fi] = beta[fi] * (2 * len(facts) + 1)
    deg_formula_err = float((deg - torch.cat([dv, de, df])).abs().max())
    h0 = torch.stack([math.sqrt(float(deg[v])) * a_vecs[labels[v]] for v in range(n)])
    xstar, min_eig = anchored_solve(L, deg, n, h0, tau, d)
    scores = xstar @ torch.stack(a_vecs).T
    acc = float((scores.argmax(1) == torch.tensor(labels)).double().mean())
    q_orth = max(float((tr.T @ tr - torch.eye(d)).abs().max()) for ts in transports for tr in ts)
    q_det = min(float(torch.det(tr)) for ts in transports for tr in ts)
    return {
        "case": name, "d": d, "m": m, "nodes": n, "facts": len(edges), "faces": len(faces),
        "classes": len(a_vecs), "label_disagreement_on_facts": sum(labels[u] != labels[v] for u, v, _ in edges) / len(edges),
        "energy_at_section": energy, "max_abs_LY": ly,
        "degree_blocks_scalar_dev": scalar_dev, "degree_formula_err": deg_formula_err,
        "min_eig_anchored_matrix": min_eig, "max_abs_xstar_minus_h0": float((xstar - h0).abs().max()),
        "argmax_accuracy": acc, "transport_orth_err": q_orth, "transport_min_det": q_det,
    }


def main():
    results = []
    d, m = 8, 4
    graph = build_graph()
    types, relations, edges, faces = graph
    n = len(types)
    C = 5
    labels = [random.randrange(C) for _ in range(n)]
    u_r = orthonormal(d, m)
    theta = [2 * math.pi * c / C for c in range(C)]
    a_circle = [math.cos(t) * u_r[:, 0] + math.sin(t) * u_r[:, 1] for t in theta]
    J = gen(m)
    frames_id = {r: torch.eye(d) for r in relations}

    def coords_circle(ei):
        u, v, _ = edges[ei]
        return u_r, (theta[labels[u]] - theta[labels[v]]) * J

    # 1. construction of Thm 4 (1.25)
    results.append(run_case("thm4_construction", graph, labels, d, m, coords_circle, frames_id, a_circle))

    # 2. stored orientation of rho(r) = r facts does not matter (same L)
    flipped = [(v, u, r) if relations[r][2] else (u, v, r) for u, v, r in edges]
    def coords_flipped(ei):
        u, v, _ = flipped[ei]
        return u_r, (theta[labels[u]] - theta[labels[v]]) * J
    a_w = [0.7] * len(edges); b_w = [0.3] * len(faces)
    L1, _ = build_L(n, edges, faces, make_maps(edges, relations, d, frames_id, coords_circle), a_w, b_w, d)
    L2, _ = build_L(n, flipped, faces, make_maps(flipped, relations, d, frames_id, coords_flipped), a_w, b_w, d)
    results.append({"case": "shared_role_orientation_invariance", "max_abs_L_diff": float((L1 - L2).abs().max())})

    # 3. corollary: given unit vectors spanning k = m dims
    Cg = 6
    a_given = [u_r @ (lambda z: z / z.norm())(torch.randn(m)) for _ in range(Cg)]
    labels_g = [random.randrange(Cg) for _ in range(n)]

    def plane_log(bs, bt):
        """B with exp(B) bt = bs, rotation in span(bt, bs); B(t,s) = -B(s,t)."""
        w1 = bt
        w2 = bs - (bs @ bt) * bt
        if w2.norm() < 1e-12:
            return torch.zeros(m, m)
        w2 = w2 / w2.norm()
        phi = math.atan2(float(bs @ w2), float(bs @ w1))
        return phi * (torch.outer(w2, w1) - torch.outer(w1, w2))

    b = [u_r.T @ x for x in a_given]
    B = {(s, t): plane_log(b[s], b[t]) for s in range(Cg) for t in range(Cg)}
    antisym = max(float((B[(s, t)] + B[(t, s)]).abs().max()) for s in range(Cg) for t in range(Cg))

    def coords_given(ei):
        u, v, _ = edges[ei]
        return u_r, B[(labels_g[u], labels_g[v])]

    res = run_case("corollary_given_vectors_k_eq_m", graph, labels_g, d, m, coords_given, frames_id, a_given)
    res["B_antisymmetry_err"] = antisym
    results.append(res)

    # 4. negative control A: one fact's rotation off by 0.3 rad -> not a section
    def perturb(maps):
        ei = next(i for i, (u, v, r) in enumerate(edges) if not relations[r][2] and labels[u] != labels[v])
        src, tgt = maps[ei]
        maps = list(maps)
        maps[ei] = (src, tgt @ exp_lowrank(u_r, 0.3 * J))
        return maps
    results.append(run_case("negative_control_perturbed_fact", graph, labels, d, m, coords_circle, frames_id, a_circle, perturb=perturb))

    # 5. negative control B: identity maps (graph diffusion) with the same heterophilic labels
    results.append(run_case("negative_control_identity_maps", graph, labels, d, m,
                            lambda ei: (u_r, torch.zeros(m, m)), frames_id, a_circle))

    # 6. map-level checks at production size: d = 64, m = 4, C = 349 (ogbn-mag), random frames
    d6, m6, C6 = 64, 4, 349
    u6 = orthonormal(d6, m6)
    th6 = [2 * math.pi * c / C6 for c in range(C6)]
    av6 = torch.stack([math.cos(t) * u6[:, 0] + math.sin(t) * u6[:, 1] for t in th6])
    J6 = gen(m6)
    resid_dist, resid_shared = 0.0, 0.0
    for _ in range(2000):
        s, t = random.randrange(C6), random.randrange(C6)
        a6 = u6 @ ((th6[s] - th6[t]) * J6) @ u6.T
        tgt = exp_lowrank(u6, (th6[s] - th6[t]) * J6)
        resid_dist = max(resid_dist, float((av6[s] - tgt @ av6[t]).abs().max()))
        resid_shared = max(resid_shared, float((torch.linalg.matrix_exp(-a6 / 2) @ av6[s]
                                                - torch.linalg.matrix_exp(a6 / 2) @ av6[t]).abs().max()))
    lam = 0.5 + 50 * torch.rand(C6)
    acc6 = float(((lam[:, None] * av6) @ av6.T).argmax(1).eq(torch.arange(C6)).double().mean())
    # closed form vs dense exp, random cores; SO(d) of R_r exp(.) with random frames
    cf, orth, mindet = 0.0, 0.0, 1.0
    for _ in range(200):
        c = torch.randn(m6, m6); c = c - c.T
        cf = max(cf, float((exp_lowrank(u6, c) - torch.linalg.matrix_exp(u6 @ c @ u6.T)).abs().max()))
        q = random_so(d6) @ exp_lowrank(u6, c)
        orth = max(orth, float((q.T @ q - torch.eye(d6)).abs().max()))
        mindet = min(mindet, float(torch.det(q)))
    results.append({"case": "production_size_d64_m4_C349", "max_resid_distinct_roles": resid_dist,
                    "max_resid_shared_roles": resid_shared, "argmax_accuracy_scaled": acc6,
                    "min_cos_gap": 1 - math.cos(2 * math.pi / C6),
                    "closed_form_vs_matrix_exp": cf, "Q_orth_err": orth, "Q_min_det": mindet})

    print(json.dumps(results, indent=1))
    ok = (results[0]["energy_at_section"] < TOL and results[0]["max_abs_xstar_minus_h0"] < 1e-8
          and results[0]["argmax_accuracy"] == 1.0 and results[1]["max_abs_L_diff"] < TOL
          and results[2]["energy_at_section"] < TOL and results[2]["argmax_accuracy"] == 1.0
          and results[3]["energy_at_section"] > 1e-3 and results[4]["energy_at_section"] > 1e-3
          and results[5]["max_resid_distinct_roles"] < 1e-10 and results[5]["argmax_accuracy_scaled"] == 1.0)
    print("ALL_CHECKS_PASS" if ok else "CHECK_FAILED")


if __name__ == "__main__":
    main()
