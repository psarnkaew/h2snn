import numpy as np
rng=np.random.default_rng(11); d=4
def orth(): q,_=np.linalg.qr(rng.normal(size=(d,d))); return q
N=20; edges=[]
while len(edges)<40:
    u,v=rng.choice(N,2,replace=False)
    if (u,v) not in edges and (v,u) not in edges: edges.append((u,v))
E=len(edges)
# consistent sheaf: F_{v<e} = R_e Q_v  -> node global sections exist (x_v = Q_v^T c)
Q=[orth() for _ in range(N)]; R=[orth() for _ in range(E)]
F={(z,k):R[k]@Q[z] for k,(u,v) in enumerate(edges) for z in (u,v)}
adj={}
for k,(u,v) in enumerate(edges): adj.setdefault(u,[]).append((k,v)); adj.setdefault(v,[]).append((k,u))
faces=[]
while len(faces)<15:
    a=rng.integers(N); path=[a]; pe=[]
    for _ in range(rng.integers(2,5)):
        nb=[(k,w) for k,w in adj.get(path[-1],[]) if w not in path]
        if not nb: break
        k,w=nb[rng.integers(len(nb))]; pe.append(k); path.append(w)
    if len(pe)>=2: faces.append((path,pe))
I=len(faces); w=np.exp(rng.normal(size=I))
# Hasse-diagram sheaf: incidences (v<e) with map F, (e<f) with map F_{e<f} (transport, A_f=I); energy sum w ||F y_low - y_high||^2
n=(N+E+I)*d; L=np.zeros((n,n))
def blk(c): return slice(c*d,(c+1)*d)
def add(lo,hi,Fm,wt):
    a,b=blk(lo),blk(hi)
    L[a,a]+=wt*Fm.T@Fm; L[b,b]+=wt*np.eye(d); L[a,b]-=wt*Fm.T; L[b,a]-=wt*Fm
for k,(u,v) in enumerate(edges):
    for z in (u,v): add(z,N+k,F[(z,k)],1.0)
for i,(path,pe) in enumerate(faces):
    T=np.eye(d)
    for j,k in enumerate(pe):
        Fe=T@F[(path[j],k)].T; add(N+k,N+E+i,Fe,w[i]); T=Fe@F[(path[j+1],k)]
ev,V=np.linalg.eigh(L)
print("Hasse Laplacian: symmetric",abs(L-L.T).max(),"| min eig",ev.min().round(12),"| dim ker",(ev<1e-9).sum(),"(node global sections: d =",d,")")
# kernel node part = global sections?
Vk=V[:,ev<1e-9]; Xn=Vk[:N*d]
D0=np.zeros((E*d,N*d))
for k,(u,v) in enumerate(edges): D0[blk(k),blk(v)]+=F[(v,k)]; D0[blk(k),blk(u)]-=F[(u,k)]
print("kernel node parts are global sections: |delta0 x| =",abs(D0@Xn).max())
# eliminate edge+face states (Schur complement onto nodes) and compare with Delta0/2 when no faces
Dg=np.diag(L).copy()
def schur(L):
    A=L[:N*d,:N*d]; Bm=L[:N*d,N*d:]; C=L[N*d:,N*d:]; return A-Bm@np.linalg.solve(C,Bm.T)
L0=L.copy(); L0[(N+E)*d:,:]=0; L0[:,(N+E)*d:]=0
# rebuild without faces
Lnf=np.zeros(((N+E)*d,)*2)
for k,(u,v) in enumerate(edges):
    for z in (u,v):
        a,b=blk(z),blk(N+k); Fm=F[(z,k)]
        Lnf[a,a]+=Fm.T@Fm; Lnf[b,b]+=np.eye(d); Lnf[a,b]-=Fm.T; Lnf[b,a]-=Fm
S0=Lnf[:N*d,:N*d]-Lnf[:N*d,N*d:]@np.linalg.solve(Lnf[N*d:,N*d:],Lnf[N*d:,:N*d])
print("no faces: Schur onto nodes == Delta0/2 :",abs(S0-0.5*D0.T@D0).max())
S=schur(L); print("with faces: Schur onto nodes PSD:",np.linalg.eigvalsh(S).min().round(12),"| equals Delta0/2 ?",abs(S-0.5*D0.T@D0).max().round(3))
# normalized spectrum bound
deg=np.zeros(N+E+I)
for k,(u,v) in enumerate(edges): deg[u]+=1; deg[v]+=1; deg[N+k]+=2
for i,(path,pe) in enumerate(faces):
    for k in pe: deg[N+k]+=w[i]; deg[N+E+i]+=w[i]
Dm=np.kron(np.diag(deg**-0.5),np.eye(d)); print("normalized Hasse Laplacian max eig:",np.linalg.eigvalsh(Dm@L@Dm).max().round(4),"(<= 2)")
