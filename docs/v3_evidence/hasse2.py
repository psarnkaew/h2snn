import numpy as np
src=open("hasse.py").read().split("ev,V=np.linalg.eigh(L)")[0]
src=src.replace("while len(faces)<15:","while len(faces)<1:")
exec(src)
D0=np.zeros((E*d,N*d))
def blk(c): return slice(c*d,(c+1)*d)
for k,(u,v) in enumerate(edges): D0[blk(k),blk(v)]+=F[(v,k)]; D0[blk(k),blk(u)]-=F[(u,k)]
S=L[:N*d,:N*d]-L[:N*d,N*d:]@np.linalg.solve(L[N*d:,N*d:],L[N*d:,:N*d])
extra=S-0.5*D0.T@D0
path,pe=faces[0]; T={path[0]:np.eye(d)}
for j,k in enumerate(pe): T[path[j+1]]=T[path[j]]@F[(path[j],k)].T@F[(path[j+1],k)]
# v2 hub Laplacian of the cell: E=1/2 sum_v ||T_v x_v - mean||^2
Sz=len(path); Lf=np.zeros((N*d,N*d))
for a in path:
    for b in path:
        Lf[blk(a),blk(b)]+=((1.0 if a==b else 0.0)-1.0/Sz)*T[a].T@T[b]
coef=(extra*Lf).sum()/(Lf*Lf).sum()
print("path length",len(pe),"| best scalar fit of (Schur - Delta0/2) to v2 hub L_f: coef",coef.round(4),"| residual",abs(extra-coef*Lf).max().round(4),"| |extra|",abs(extra).max().round(4))
print("extra touches nodes outside the cell:",sorted({i//d for i in np.where(abs(extra).sum(1)>1e-9)[0]})," cell nodes:",sorted(path))
