import numpy as np
rng=np.random.default_rng(3); d=4
def orth(): q,_=np.linalg.qr(rng.normal(size=(d,d))); return q
# random graph: 30 nodes, 60 edges, 25 faces = simple paths of length 2-4 along edges
N=30; edges=[]
while len(edges)<60:
    u,v=rng.choice(N,2,replace=False)
    if (u,v) not in edges and (v,u) not in edges: edges.append((u,v))
E=len(edges); F={(z,k):orth() for k,(u,v) in enumerate(edges) for z in (u,v)}
adj={}
for k,(u,v) in enumerate(edges): adj.setdefault(u,[]).append((k,v)); adj.setdefault(v,[]).append((k,u))
faces=[]
while len(faces)<25:
    a=rng.integers(N); path=[a]; pe=[]; L=rng.integers(2,5)
    for _ in range(L):
        nb=[(k,w) for k,w in adj.get(path[-1],[]) if w not in path]
        if not nb: break
        k,w=nb[rng.integers(len(nb))]; pe.append(k); path.append(w)
    if len(pe)>=2: faces.append((path,pe))
I=len(faces); w=np.exp(rng.normal(scale=2,size=I))     # unbounded positive WISP-like weights
S=np.zeros((E*d,N*d)); B=np.zeros((I*d,E*d))
for k,(u,v) in enumerate(edges):
    for z in (u,v): S[k*d:(k+1)*d,z*d:(z+1)*d]=F[(z,k)]
for i,(path,pe) in enumerate(faces):
    T=np.eye(d)                       # A_f = I: face frame = anchor frame
    for j,k in enumerate(pe):
        Fe=T@F[(path[j],k)].T          # F^{-1}=F^T (orthogonal)
        B[i*d:(i+1)*d,k*d:(k+1)*d]=np.sqrt(w[i])*Fe; T=Fe@F[(path[j+1],k)]
n0,n1,n2=N*d,E*d,I*d; M=np.zeros((n0+n1+n2,)*2)
M[n0:n0+n1,:n0]=S; M[:n0,n0:n0+n1]=S.T; M[n0+n1:,n0:n0+n1]=B; M[n0:n0+n1,n0+n1:]=B.T
# scalar weighted degrees from block norms
a=np.zeros((N+E+I,)*2)
for k,(u,v) in enumerate(edges): a[N+k,u]=a[u,N+k]=a[N+k,v]=a[v,N+k]=1
for i,(path,pe) in enumerate(faces):
    for k in pe: a[N+E+i,N+k]=a[N+k,N+E+i]=np.sqrt(w[i])
deg=a.sum(1); deg[deg==0]=1; Dm=np.kron(np.diag(deg**-0.5),np.eye(d))
Mn=Dm@M@Dm
print("symmetric:",abs(Mn-Mn.T).max(),"| spectral norm:",np.abs(np.linalg.eigvalsh(Mn)).max().round(6),"| I =",I,"| w range",w.min().round(3),w.max().round(1))
