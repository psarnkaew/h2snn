import numpy as np
exec(open("v3_dyn.py").read().split("L1=D0")[0])  # reuse graph, maps, D0, D1, P
W1=np.random.default_rng(5).normal(size=(d,2*d))
def phi(k,x):  # nonlinear two-sided edge update, edge k=(u,v)
    u,v=edges[k]; a=F[(u,k)]@x[idx[u]*d:(idx[u]+1)*d]; b=F[(v,k)]@x[idx[v]*d:(idx[v]+1)*d]
    return np.tanh(W1@np.concatenate([a,b]))
def one_layer(x,s,phi_on):
    s=s.copy()
    for k in phi_on: s[k*d:(k+1)*d]+=phi(k,x)
    return P@(D1.T@(D1@s))   # lift of face term into nodes (observed edges only)
rng=np.random.default_rng(2); x=rng.normal(size=N*d); s=rng.normal(size=E*d)
dx=np.zeros(N*d); dx[idx["P1"]*d:(idx["P1"]+1)*d]=rng.normal(size=d)  # perturb P1 only
for name,on in [("phi on closures only",[3,4]),("phi on observed edges only",[0,1,2])]:
    ch=one_layer(x+dx,s,on)-one_layer(x,s,on)
    print(name,"| change at P9:",abs(ch[idx["P9"]*d:(idx["P9"]+1)*d]).max().round(3))
