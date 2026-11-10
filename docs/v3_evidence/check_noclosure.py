import numpy as np, importlib.util
spec = importlib.util.spec_from_file_location("cf", "check_face.py")
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    cf = importlib.util.module_from_spec(spec); spec.loader.exec_module(cf)
d, idx, nodes = cf.d, cf.idx, cf.nodes
D0, D1, P = cf.build(cf.A)
# lift from observed edges only: zero the closure-edge block (edge id 3)
P_obs = P.copy(); P_obs[:, 3*d:4*d] = 0
M = P_obs @ D1.T
for n in nodes:
    print(f"lift without closure edge, face->{n}:", np.abs(M[idx[n]*d:(idx[n]+1)*d]).max().round(4))
# v1 node operator L0 over E_ext contains the closure edge = a direct P1-P9 coupling
L0 = D0.T @ D0
print("v1 L0 block P1-P9 (via closure edge):", np.abs(L0[idx['P1']*d:(idx['P1']+1)*d, idx['P9']*d:(idx['P9']+1)*d]).max().round(4))
D0_obs = D0.copy(); D0_obs[3*d:4*d, :] = 0
L0o = D0_obs.T @ D0_obs
print("L0 over observed edges only, P1-P9 block:", np.abs(L0o[idx['P1']*d:(idx['P1']+1)*d, idx['P9']*d:(idx['P9']+1)*d]).max())
