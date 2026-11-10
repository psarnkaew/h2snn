# Idealised S1-prime-B check (design session 2026-11-11): d = 1, identity maps, a_e = 1, exact sparse solve of (P^T P + tau L_hat) Y = P^T h on the three-level energy; full vs covering_only. Not a model run.
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
def run(n,m,k,beta,node_cell,tau=1.0,seed=0):
    rng=np.random.default_rng(seed)
    y=rng.choice([-1.,1.],n)
    # nodes: P (n), A (n*m), Q (n*m*k)
    nP,nA,nQ=n,n*m,n*m*k; N=nP+nA+nQ
    A_of=np.repeat(np.arange(n),m)            # author -> paper
    Q_of=np.repeat(np.arange(nA),k)           # cpaper -> author
    hQ=rng.choice([-1.,1.],nQ)
    rel=np.zeros(nQ,bool); pick=rng.integers(0,m*k,n); idx=np.arange(n)*m*k+pick; rel[idx]=True; hQ[idx]=y
    h=np.concatenate([np.zeros(nP),np.zeros(nA),hQ])
    # facts: PA (nA), AQ (nQ)
    E=nA+nQ; F=nQ
    eu=np.concatenate([A_of, nP+Q_of]); ev=np.concatenate([nP+np.arange(nA), nP+nA+np.arange(nQ)])
    bf=np.where(rel,np.exp(beta),1.0); a=1.0
    rows=[];cols=[];vals=[]
    def term(i,j,w):  # w/2 (y_i - y_j)^2
        rows.extend([i,j,i,j]);cols.extend([i,j,j,i]);vals.extend([w,w,-w,-w])
    S=N; Z=N+E
    for e in range(E):
        term(eu[e],S+e,a); term(ev[e],S+e,a)
    for f in range(F):
        Aq=Q_of[f]; Pp=A_of[Aq]; ePA=Aq; eAQ=nA+f
        for e in (ePA,eAQ): term(S+e,Z+f,bf[f])
        if node_cell:
            for v in (Pp,nP+Aq,nP+nA+f): term(v,Z+f,bf[f])
    T=N+E+F
    L=sp.csr_matrix((vals,(rows,cols)),shape=(T,T))
    D=L.diagonal(); Dm=sp.diags(1/np.sqrt(D))
    Lh=Dm@L@Dm
    Pm=sp.diags(np.concatenate([np.ones(N),np.zeros(E+F)]))
    rhs=np.concatenate([h,np.zeros(E+F)])
    Y=sla.spsolve((Pm+tau*Lh).tocsc(),rhs)
    xP=Y[:nP]
    # relevant cell share of D_P (node-level degree incl. cell terms)
    return float(np.mean(np.sign(xP)==y)), float(np.corrcoef(xP,y)[0,1]), float(np.abs(xP).mean())
for (m,k) in [(8,8),(32,2)]:
    for beta in [0,np.log(64),np.log(512),np.log(4096)]:
        print(f"m={m:2d} k={k} beta=e^{np.exp(beta):6.0f}  full={run(400,m,k,beta,True)}  cov={run(400,m,k,beta,False)}")
