import numpy as np
exec(open("v3a_norm.py").read().replace('print("symmetric"','_=("symmetric"'))
C=N+E+I; rg=np.random.default_rng(7)
g=rg.uniform(0,1,C); b=rg.uniform(0,1,C); g[0]=0.0; b[1]=0.0     # node 0 never listens, node 1 never broadcasts
G=np.kron(np.diag(g),np.eye(d)); Bm=np.kron(np.diag(b),np.eye(d))
A=G@Mn@Bm
ev=np.linalg.eigvals(A)
print("asymmetry |A-A^T|:",abs(A-A.T).max().round(3))
print("max |Im eig|:",abs(ev.imag).max(),"| spectral radius:",abs(ev).max().round(3),"<= max g*b:",(g*b).max().round(3))
for j in (2,4):
    Aj=np.linalg.matrix_power(A,j)
    print(f"A^{j}: row of non-listening node 0 =",abs(Aj[:d]).max(),"| column of non-broadcasting node 1 =",abs(Aj[:,d:2*d]).max())
