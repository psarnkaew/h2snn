"""Is v3-B (node-fact-cell energy) = v2 hub A with participation kappa, after eliminating fact and cell states?
Energy B: 1/2 sum_{e on cells} a_e sum_{v in e} ||F_ve x_v - s_e||^2 + 1/4 sum_{e off} a_e||F_u x_u - F_v x_v||^2
        + 1/2 sum_f beta_f sum_i ||R_{e_i,f} s_{e_i} - z_f||^2
Claim (no fact shared by two cells): node-level Schur complement = 
   sum_{e on cell f} 1/2 * a^2/(2a+beta) ||F_u x_u - F_v x_v||^2  +  sum_f 1/2 sum_{v in f} kappa_v ||T_v x_v - z_f||^2 (z_f eliminated)
   kappa_v = sum_{e in f, e ni v} a_e beta_f/(2 a_e + beta_f)."""
import numpy as np
rng=np.random.default_rng(0); d=3
def orth(): return np.linalg.qr(rng.normal(size=(d,d)))[0]
def run(nodes, edges, cells):
    N=len(nodes); idx={n:i for i,n in enumerate(nodes)}; E=len(edges)
    F={(n,k):orth() for k,(u,v) in enumerate(edges) for n in (u,v)}
    a=np.exp(rng.normal(size=E)); beta=np.exp(rng.normal(size=len(cells)))
    on={k for _,pe in cells for k in pe}; onl=sorted(on); pos={k:i for i,k in enumerate(onl)}
    nS=len(onl); n=(N+nS+len(cells))*d; L=np.zeros((n,n))
    B=lambda c: slice(c*d,(c+1)*d)
    def add(i,j,Fi,Fj,w):
        L[B(i),B(i)]+=w*Fi.T@Fi; L[B(j),B(j)]+=w*Fj.T@Fj; L[B(i),B(j)]-=w*Fi.T@Fj; L[B(j),B(i)]-=w*Fj.T@Fi
    I=np.eye(d)
    for k,(u,v) in enumerate(edges):
        if k in on:
            for z in (u,v): add(idx[z],N+pos[k],F[(z,k)],I,a[k])
        else: add(idx[u],idx[v],F[(u,k)],F[(v,k)],a[k]/2)
    Ts=[]
    for c,(path,pe) in enumerate(cells):
        T={path[0]:I}
        for j,k in enumerate(pe):
            R=T[path[j]]@F[(path[j],k)].T; add(N+pos[k],N+nS+c,R,I,beta[c]); T[path[j+1]]=R@F[(path[j+1],k)]
        Ts.append(T)
    nx=N*d; S_B=L[:nx,:nx]-L[:nx,nx:]@np.linalg.solve(L[nx:,nx:],L[nx:,:nx])
    # A with kappa
    LA=np.zeros((nx+len(cells)*d,)*2)
    def addA(i,j,Fi,Fj,w):
        LA[B(i),B(i)]+=w*Fi.T@Fi; LA[B(j),B(j)]+=w*Fj.T@Fj; LA[B(i),B(j)]-=w*Fi.T@Fj; LA[B(j),B(i)]-=w*Fj.T@Fi
    cellof={k:c for c,(_,pe) in enumerate(cells) for k in pe}
    for k,(u,v) in enumerate(edges):
        if k in on: c=cellof[k]; addA(idx[u],idx[v],F[(u,k)],F[(v,k)],a[k]**2/(2*a[k]+beta[c]))
        else: addA(idx[u],idx[v],F[(u,k)],F[(v,k)],a[k]/2)
    for c,(path,pe) in enumerate(cells):
        for j,k in enumerate(pe):
            kap=a[k]*beta[c]/(2*a[k]+beta[c])
            for z in (path[j],path[j+1]): addA(idx[z],N+c,Ts[c][z],I,kap)
    S_A=LA[:nx,:nx]-LA[:nx,nx:]@np.linalg.solve(LA[nx:,nx:],LA[nx:,:nx])
    return S_B,S_A,idx
nodes=["P1","A7","P9","P4","S3","P2","A8","P5"]
# case 1: two cells sharing node P1 but no fact
edges=[("P1","A7"),("A7","P9"),("P1","S3"),("S3","P2"),("P1","A8"),("A8","P5"),("A7","P4")]
cells=[(["P1","A7","P9"],[0,1]),(["P1","S3","P2"],[2,3])]
SB,SA,idx=run(nodes,edges,cells); print("no shared fact:  max|S_B - S_A(kappa)| =",abs(SB-SA).max())
# case 2: f1=P1-A7-P9, f2=P1-A7-P4 share fact e0=(P1,A7)
cells=[(["P1","A7","P9"],[0,1]),(["P1","A7","P4"],[0,6])]
SB,SA,idx=run(nodes,edges,cells); print("shared fact e0: max|S_B - S_A(kappa)| =",abs(SB-SA).max(), "(kappa formula no longer exact)")
b=lambda n: slice(idx[n]*d,(idx[n]+1)*d)
print("   direct P9-P4 block: B",abs(SB[b("P9"),b("P4")]).max().round(4)," | any hub-sum over cells {P1,A7,P9},{P1,A7,P4}: 0 by construction",abs(SA[b("P9"),b("P4")]).max())
