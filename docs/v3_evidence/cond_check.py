"""Check ChatGPT's claim: full-system (Jacobi-preconditioned) condition number vs beta,
covering-only vs with direct node-cell term; identity maps, path P1-A7-P9, tau=1, symmetric normalization."""
import numpy as np
d=1; tau=1.0
def build(beta,direct,a=1.0,norm=True):
    # unknowns: x0,x1,x2 | s0,s1 | z   (d=1, identity maps)
    n=6; L=np.zeros((n,n))
    def add(i,j,w): L[i,i]+=w; L[j,j]+=w; L[i,j]-=w; L[j,i]-=w
    add(0,3,a); add(1,3,a); add(1,4,a); add(2,4,a)      # node-edge
    add(3,5,beta); add(4,5,beta)                        # edge-cell
    if direct:
        for v in (0,1,2): add(v,5,beta)                 # node-cell
    if norm:
        D=np.diag(L).copy(); Dm=np.diag(1/np.sqrt(D)); L=Dm@L@Dm
    P=np.zeros((3,n)); P[:,:3]=np.eye(3)
    M=P.T@P+tau*L
    J=np.diag(1/np.sqrt(np.diag(M))); Mh=J@M@J
    S=M[:3,:3]-M[:3,3:]@np.linalg.solve(M[3:,3:],M[3:,:3])   # node-level (I+tau*Shat)
    return np.linalg.cond(Mh), np.linalg.cond(S)
for beta in (1,10,100,1000):
    ca,na=build(beta,False); cb,nb=build(beta,True)
    print(f"beta={beta:5}: covering-only full/Jacobi cond={ca:7.1f} node-level={na:4.2f} | with direct term full={cb:6.1f} node-level={nb:4.2f}")
