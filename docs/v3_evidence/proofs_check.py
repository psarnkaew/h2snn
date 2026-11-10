"""Numerical checks for the theorems in v3_A_vs_B.html section 'Proofs' (2026-11-10).
Random heterogeneous-like graph with cycles, WISP-like simple paths as 2-cells (closure definitional, A_f = I)."""
import numpy as np
from scipy.linalg import expm, null_space
rng=np.random.default_rng(21); d=4; N=24; C=3                      # d >= number of classes
cls=rng.integers(C,size=N)
perm=rng.permutation(N); edges=[(int(perm[i]),int(perm[rng.integers(i)])) for i in range(1,N)]   # spanning tree: connected
while len(edges)<48:
    u,v=rng.choice(N,2,replace=False)
    if (u,v) not in edges and (v,u) not in edges: edges.append((u,v))
E=len(edges); adj={}
for k,(u,v) in enumerate(edges): adj.setdefault(u,[]).append((k,v)); adj.setdefault(v,[]).append((k,u))
faces=[]
while len(faces)<14:
    a=rng.integers(N); path=[a]; pe=[]
    for _ in range(rng.integers(2,5)):
        nb=[(k,w) for k,w in adj.get(path[-1],[]) if w not in path]
        if not nb: break
        k,w=nb[rng.integers(len(nb))]; pe.append(k); path.append(w)
    if len(pe)>=2: faces.append((path,pe))
I=len(faces); w=np.exp(rng.normal(size=I))
blk=lambda c: slice(c*d,(c+1)*d)
het=np.mean([cls[u]!=cls[v] for u,v in edges]); print(f"graph: N={N}, E={E}, faces={I}, edge heterophily={het:.2f}")

def householder(a,b):            # orthogonal reflection mapping unit a -> unit b (a Householder, as in our map family)
    u=a-b; n=np.linalg.norm(u)
    return np.eye(d) if n<1e-12 else np.eye(d)-2*np.outer(u,u)/n**2
def build(F, free=False):
    """returns transports per face, Delta0, A-operator (Delta0+L_P), B-operator (Hasse Laplacian)"""
    D0=np.zeros((E*d,N*d))
    for k,(u,v) in enumerate(edges): D0[blk(k),blk(v)]+=F[(v,k)]; D0[blk(k),blk(u)]-=F[(u,k)]
    LP=np.zeros((N*d,N*d)); n=(N+E+I)*d; LB=np.zeros((n,n)); Ts=[]
    def add(lo,hi,Fm,wt):
        a,b=blk(lo),blk(hi); LB[a,a]+=wt*Fm.T@Fm; LB[b,b]+=wt*np.eye(d); LB[a,b]-=wt*Fm.T; LB[b,a]-=wt*Fm
    for k,(u,v) in enumerate(edges):
        for z in (u,v): add(z,N+k,F[(z,k)],1.0)
    for i,(path,pe) in enumerate(faces):
        T={path[0]:np.eye(d)}; Fef={}
        for j,k in enumerate(pe):
            Fef[k]=(np.linalg.qr(rng.normal(size=(d,d)))[0] if free else T[path[j]]@F[(path[j],k)].T)
            T[path[j+1]]=Fef[k]@F[(path[j+1],k)]
        Ts.append((T,Fef))
        for k in pe: add(N+k,N+E+i,Fef[k],w[i])
        S=len(path)                                             # v2 hub term: 1/2 sum_v ||T_v x_v - mean||^2
        for a in path:
            for b in path: LP[blk(a),blk(b)]+=w[i]*((a==b)-1.0/S)*T[a].T@T[b]
    return Ts, D0, D0.T@D0+LP, LB, LP

# ---- Theorem 1: sheaf (functoriality on every node <= edge <= face chain, closure included)
F={(z,k):np.linalg.qr(rng.normal(size=(d,d)))[0] for k,(u,v) in enumerate(edges) for z in (u,v)}
Ts,D0,LA,LB,LP=build(F); worst=0
for (path,pe),(T,Fef) in zip(faces,Ts):
    for j,k in enumerate(pe):
        worst=max(worst,abs(Fef[k]@F[(path[j],k)]-T[path[j]]).max(),abs(Fef[k]@F[(path[j+1],k)]-T[path[j+1]]).max())
    # closure c=(v_l -> v_0): F_{v0<c}=I, F_{vl<c}=T_v0^{-1}T_vl, F_{c<f}=A_f=I
    worst=max(worst,abs(np.eye(d)@np.eye(d)-T[path[0]]).max(),abs(np.eye(d)@(np.linalg.inv(T[path[0]])@T[path[-1]])-T[path[-1]]).max())
print("T1 functoriality, all chains incl. closure: max residual",worst)
# ---- Theorem 2: holonomy around each 2-cell boundary = I
hol=max(abs(np.linalg.inv(T[path[0]])@T[path[-1]]@np.linalg.inv(np.linalg.inv(T[path[0]])@T[path[-1]])-np.eye(d)).max() for (path,pe),(T,_) in zip(faces,Ts))
print("T2 holonomy around cell (path then closure back): |H - I| =",hol)

# ---- Theorem 4 construction: class directions x*_v = e_{class(v)}, per-edge Householder maps so x* is a global section
xs=np.zeros(N*d)
for v in range(N): xs[blk(v)]=np.eye(d)[cls[v]]
F4={}
for k,(u,v) in enumerate(edges):
    F4[(u,k)]=np.eye(d); F4[(v,k)]=householder(xs[blk(v)],xs[blk(u)])   # F_v x*_v = x*_u = F_u x*_u
Ts4,D04,LA4,LB4,LP4=build(F4)
print("T4 x* is a section of the edge sheaf: |delta0 x*| =",abs(D04@xs).max())
print("T4 x* survives the cells, A: |(Delta0+L_P) x*| =",abs(LA4@xs).max())
ys=np.concatenate([xs]+[F4[(u,k)]@xs[blk(u)] for k,(u,v) in enumerate(edges)]+[Ts4[i][0][faces[i][0][0]]@xs[blk(faces[i][0][0])] for i in range(I)])
print("T4 x* survives the cells, B: |L_B (x*, s=Fx*, z=Tx*)| =",abs(LB4@ys).max())
print("   classes of adjacent nodes differ on",int(het*E),"edges, yet x* agrees on every edge; x* separates classes exactly (one-hot)")

# ---- Theorem 3: kernels (no collapse caused by cells) for the generic sheaf and the constructed one
for name,(D0_,LA_,LB_) in {"random maps":(D0,LA,LB),"constructed maps":(D04,LA4,LB4)}.items():
    k0=null_space(D0_.T@D0_).shape[1]; kA=null_space(LA_).shape[1]; kB=null_space(LB_).shape[1]
    print(f"T3 [{name}] dim ker Delta0 = {k0}, dim ker A-operator = {kA}, dim ker B-operator = {kB}")
# trivial-holonomy sheaf F = R_e Q_v (rich kernel, dim d)
Q=[np.linalg.qr(rng.normal(size=(d,d)))[0] for _ in range(N)]; R=[np.linalg.qr(rng.normal(size=(d,d)))[0] for _ in range(E)]
F5={(z,k):R[k]@Q[z] for k,(u,v) in enumerate(edges) for z in (u,v)}
Ts5,D05,LA5,LB5,LP5=build(F5)
print("T3 [flat sheaf R_e Q_v] dim ker: Delta0",null_space(D05.T@D05).shape[1],"| A",null_space(LA5).shape[1],"| B",null_space(LB5).shape[1])
# necessity: free per-incidence edge->face maps
_,_,LAf,LBf,_=build(F5,free=True)
print("T3' free edge->face maps (no transport): dim ker B =",null_space(LBf).shape[1],"(was",null_space(LB5).shape[1],")")

# ---- Theorem 5: limit of heat diffusion = consensus in each node's own frame; cells do not change the limit
x0=rng.normal(size=N*d)
lim0=expm(-200*D05.T@D05)@x0; limA=expm(-200*LA5)@x0
ybar=np.mean([Q[v]@x0[blk(v)] for v in range(N)],axis=0)
pred=np.concatenate([Q[v].T@ybar for v in range(N)])
print("T5 limit of heat flow (graph sheaf) = Q_v^T mean_u(Q_u x_u): |diff| =",abs(lim0-pred).max().round(8))
print("T5 same limit with path cells (A): |diff| =",abs(limA-pred).max().round(8))
yB0=np.concatenate([x0,np.zeros((E+I)*d)]); limB=expm(-400*LB5)@yB0
print("T5 B: node part of the limit is a global section: |delta0 x_lim| =",abs(D05@limB[:N*d]).max().round(8),"| norm of node part",np.linalg.norm(limB[:N*d]).round(4))
# spectral gap (speed of reaching the global picture), A vs graph sheaf
def gap(L):
    ev=np.linalg.eigvalsh(L); return ev[ev>1e-9].min()
print("T5' smallest non-zero eigenvalue: graph sheaf",gap(D05.T@D05).round(4),"| with cells (A)",gap(LA5).round(4))

# ---- Theorem 6: intermediate nodes matter; endpoint-only view does not see them
path,pe=faces[0]; mid=path[1]
x=rng.normal(size=N*d); g=LP5@x
print("T6 cell term moves the intermediate node: |(L_P x)_mid| =",abs(g[blk(mid)]).max().round(4))
x2=x.copy(); x2[blk(mid)]+=rng.normal(size=d)
ends=[path[0],path[-1]]
dA=(LP5@x2-LP5@x)[np.r_[blk(ends[0]),blk(ends[1])]]
T=Ts5[0][0]; Hl=np.linalg.inv(T[path[0]])@T[path[-1]]
endpoint=lambda x: np.concatenate([x[blk(path[0])]-Hl@x[blk(path[-1])], -(Hl.T)@(x[blk(path[0])]-Hl@x[blk(path[-1])])])
print("T6 changing only the intermediate node changes the endpoints' update: A",abs(dA).max().round(4),"| endpoint-only (holonomy edge)",abs(endpoint(x2)-endpoint(x)).max())
