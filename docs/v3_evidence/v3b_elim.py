"""v3-B (anchored solve) checks, 2026-11-10:
(1) exact elimination: facts on no cell can be replaced by the direct fact term (1/2) c^2 ||F_u x_u - F_v x_v||^2
    without changing x* of  min 1/2||x-h||^2 + tau/2 Y^T L Y ;
(2) preconditioned Richardson (GPT sec.20) converges to the direct solve."""
import numpy as np
exec(open("proofs_check.py").read().split("# ---- Theorem 1")[0])
F={(z,k):np.linalg.qr(rng.normal(size=(d,d)))[0] for k,(u,v) in enumerate(edges) for z in (u,v)}
c2=np.exp(rng.normal(scale=.5,size=E)); tau=1.0
on_cell=sorted({k for _,pe in faces for k in pe}); off=[k for k in range(E) if k not in on_cell]
def build(keep):                       # keep = facts with an explicit edge state
    pos={k:i for i,k in enumerate(keep)}; n=(N+len(keep)+I)*d; L=np.zeros((n,n))
    def B(c): return slice(c*d,(c+1)*d)
    def add(a,b,Fa,Fb,wt):             # wt * ||Fa y_a - Fb y_b||^2 /2  (Fb may be I)
        A_,B_=B(a),B(b); L[A_,A_]+=wt*Fa.T@Fa; L[B_,B_]+=wt*Fb.T@Fb; L[A_,B_]-=wt*Fa.T@Fb; L[B_,A_]-=wt*Fb.T@Fa
    for k,(u,v) in enumerate(edges):
        if k in pos:
            for z in (u,v): add(z,N+pos[k],F[(z,k)],np.eye(d),c2[k])
        else: add(u,v,F[(u,k)],F[(v,k)],c2[k]/2)          # eliminated fact = direct term with weight c^2/2
    for i,(path,pe) in enumerate(faces):
        T=np.eye(d)
        for j,k in enumerate(pe):
            Fe=T@F[(path[j],k)].T; add(N+pos[k],N+len(keep)+i,Fe,np.eye(d),w[i]); T=Fe@F[(path[j+1],k)]
    return L
h=rng.normal(size=N*d)
def solve(L):
    n=L.shape[0]; P=np.zeros((N*d,n)); P[:,:N*d]=np.eye(N*d); M=P.T@P+tau*L
    Y=np.linalg.solve(M,P.T@h); return M,P,Y
M1,P1,Y1=solve(build(list(range(E)))); M2,P2,Y2=solve(build(on_cell))
print(f"facts {E}, on a cell {len(on_cell)}, eliminated {len(off)}")
print("x* all edge states vs only on-cell edge states: |diff| =",abs(Y1[:N*d]-Y2[:N*d]).max())
print("M positive definite: min eig =",np.linalg.eigvalsh(M1).min().round(5))
# preconditioned Richardson with J = P^T P + tau*blockdiag(L)
L=build(list(range(E))); n=L.shape[0]; Jd=np.zeros((n,n))
for c in range(n//d): Jd[c*d:(c+1)*d,c*d:(c+1)*d]=L[c*d:(c+1)*d,c*d:(c+1)*d]
J=P1.T@P1+tau*Jd; Ji=np.linalg.inv(J); Y=P1.T@h
for t in range(1,401):
    Y=Y-0.5*Ji@(M1@Y-P1.T@h)
    if t in (25,50,100,200,400): print(f"iteration {t}: |x - x*| = {abs(Y[:N*d]-Y1[:N*d]).max():.2e}")
# (3) preconditioned conjugate gradient on the same SPD system (Jacobi preconditioner J)
b=P1.T@h; Y=np.zeros_like(b); r=b-M1@Y; zt=Ji@r; p=zt.copy(); rz=r@zt
for t in range(1,61):
    Mp=M1@p; a=rz/(p@Mp); Y=Y+a*p; r=r-a*Mp; zt=Ji@r; rz2=r@zt; p=zt+(rz2/rz)*p; rz=rz2
    if t in (5,10,15,20,30,60): print(f"PCG iteration {t}: |x - x*| = {abs(Y[:N*d]-Y1[:N*d]).max():.2e}")
print("condition number of J^-1/2 M J^-1/2:",np.linalg.cond(np.linalg.cholesky(Ji).T@M1@np.linalg.cholesky(Ji)).round(1))
