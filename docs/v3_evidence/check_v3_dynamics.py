"""v3 check (2026-11-10): two instances P1-A7-P9, P1-A7-P4 sharing e0.
1) chain condition needs closure rows in delta0; 2) linear coboundary node->edge: face values independent of node states;
3) tanh on edges breaks that; 4) faces couple through shared edge e0."""
import numpy as np
rng=np.random.default_rng(0); d=3
nodes=["P1","A7","P9","P4"]; idx={n:i for i,n in enumerate(nodes)}; N=4
# e0 P1->A7, e1 A7->P9, e2 A7->P4, e3 closure f1 (P9->P1), e4 P4->? second instance P1-A7-P4 closure e4 (P4->P1)
edges=[("P1","A7"),("A7","P9"),("A7","P4"),("P9","P1"),("P4","P1")]; E=len(edges)
def conf():
    q,_=np.linalg.qr(rng.normal(size=(d,d))); return rng.uniform(.5,2)*q
F={}
for k,(u,v) in enumerate(edges[:3]): F[(u,k)]=conf(); F[(v,k)]=conf()
faces=[(["P1","A7","P9"],[0,1],3),(["P1","A7","P4"],[0,2],4)]
D1=np.zeros((len(faces)*d,E*d))
for fi,(pn,pe,cl) in enumerate(faces):
    A=rng.normal(size=(d,d)); T={pn[0]:A}
    for i,e in enumerate(pe):
        u,v=pn[i],pn[i+1]; Fe=T[u]@np.linalg.inv(F[(u,e)]); D1[fi*d:(fi+1)*d,e*d:(e+1)*d]=Fe; T[v]=Fe@F[(v,e)]
    F[(pn[0],cl)]=np.eye(d); F[(pn[-1],cl)]=np.linalg.inv(T[pn[0]])@T[pn[-1]]
    D1[fi*d:(fi+1)*d,cl*d:(cl+1)*d]=A
D0=np.zeros((E*d,N*d))
for k,(u,v) in enumerate(edges):
    D0[k*d:(k+1)*d,idx[v]*d:(idx[v]+1)*d]+=F[(v,k)]; D0[k*d:(k+1)*d,idx[u]*d:(idx[u]+1)*d]-=F[(u,k)]
print("chain cond with closures in delta0:",abs(D1@D0).max())
D0nc=D0.copy(); D0nc[3*d:,:]=0
print("chain cond if closure rows dropped from delta0:",abs(D1@D0nc).max().round(3))
# lift through observed edges only
P=np.zeros((N*d,E*d))
for k,(u,v) in enumerate(edges[:3]):
    for z in (u,v): P[idx[z]*d:(idx[z]+1)*d,k*d:(k+1)*d]+=F[(z,k)].T
L1=D0@D0.T+D1.T@D1
def run(x0,s0,K=4,a=.05,b=.3,nonlin=False):
    x,s=x0.copy(),s0.copy(); ys=[]
    for _ in range(K):
        s_new=s-a*(L1@s)+b*(D0@x)            # coboundary node->edge
        x_new=x-a*(D0nc.T@D0nc@x)+b*(P@(D1.T@(D1@s)))  # lift of face term, observed edges
        if nonlin: s_new=np.tanh(s_new)
        x,s=x_new,s_new; ys.append(D1@s)
    return np.array(ys)
s0=rng.normal(size=E*d); xa=rng.normal(size=N*d); xb=rng.normal(size=N*d)
print("linear: face values with node state xa vs xb (max diff over 4 layers):",abs(run(xa,s0)-run(xb,s0)).max())
print("tanh on edges: same comparison:",abs(run(xa,s0,nonlin=True)-run(xb,s0,nonlin=True)).max().round(3))
# instance coupling via shared edge e0: perturb face-1 input only through edge e1, see face-2 change
y=D1@D1.T; print("face1-face2 coupling block of delta1 delta1^T (shared e0):",abs(y[:d,d:]).max().round(3))
# A_f scalar redundant with w_f
