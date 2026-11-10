import numpy as np
exec(open("v3_dyn.py").read().split("L1=D0")[0])   # graph, F (incl. closure maps), D0 (signed), D1, P (unsigned lift, observed)
# rebuild transports of f1 to test functoriality / route independence at A7
rng2=np.random.default_rng(9)
def Fef(face):  # edge->face blocks of face index
    return {e:D1[face*d:(face+1)*d,e*d:(e+1)*d] for e in range(E) if abs(D1[face*d:(face+1)*d,e*d:(e+1)*d]).max()>0}
B=Fef(0)  # f1: e0,e1,c1(=3)
print("functoriality at A7: |F_e0f F_A7e0 - F_e1f F_A7e1| =",abs(B[0]@F[("A7",0)]-B[1]@F[("A7",1)]).max())
print("functoriality at P1: |F_e0f F_P1e0 - F_c1f F_P1c1| =",abs(B[0]@F[("P1",0)]-B[3]@F[("P1",3)]).max())
print("functoriality at P9: |F_e1f F_P9e1 - F_c1f F_P9c1| =",abs(B[1]@F[("P9",1)]-B[3]@F[("P9",3)]).max())
# unsigned (sum) node->edge, all 5 edges
S=np.zeros((E*d,N*d))
for k,(u,v) in enumerate(edges):
    S[k*d:(k+1)*d,idx[u]*d:(idx[u]+1)*d]+=F[(u,k)]; S[k*d:(k+1)*d,idx[v]*d:(idx[v]+1)*d]+=F[(v,k)]
x=rng2.normal(size=N*d)
T={n:B[0]@F[(n,0)] for n in ["P1","A7"]}; T["P9"]=B[1]@F[("P9",1)]
hub=sum(T[n]@x[idx[n]*d:(idx[n]+1)*d] for n in T)
print("stateless sum edges incl closure: |face - 2*hub| =",abs((D1@S@x)[:d]-2*hub).max())
# adjoint-pair check: node->edge->node operators
Ssym=S.T@S; print("unsigned pair  S^T S symmetric:",abs(Ssym-Ssym.T).max())
Mix=P@D0; print("brief mix  P delta0 (signed up, unsigned down) asymmetry:",abs(Mix-Mix.T).max().round(3))
# face -> node through adjoints
print("face->node via signed adjoint delta0^T delta1^T:",abs(D0.T@D1.T).max(), "| via unsigned S^T delta1^T:",abs(S.T@D1.T).max().round(3))
