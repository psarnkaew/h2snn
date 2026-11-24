"""Card 21 C1 evidence: HINormer `build_node_sequences_for` on H-Pokec (len_seq 50), with the adapter's H2GB prelude
(`hinormer._fit_h2gb`: seed_everything -> homogeneous(feats_type=0) -> restore_legacy_message_functions -> dgl.graph).
Prints the sequence cache key (which holds `random.getstate()` at the call) and the call's time. Run twice in fresh
processes with the same seed and the same cache dir: equal keys = the state at the call is the same across trials
that share a seed; the second run is a hit. Usage (from H2S-NN/): python ../docs/card21_evidence/c1_hpokec_timing.py SEED CACHE_DIR
"""
import hashlib
import pickle
import random
import resource
import sys
import time

import torch

sys.path.insert(0, ".")
import dgl  # noqa: E402

from src.baselines.adapters import hinormer  # noqa: E402
from src.baselines.bundles import build_bundle  # noqa: E402
from src.baselines.runtime._dgl_compat import restore_legacy_message_functions  # noqa: E402
from src.utils.misc import seed_everything  # noqa: E402

seed, cache_dir = int(sys.argv[1]), sys.argv[2]
t = time.time()
bundle = build_bundle("node_classification", "h-pokec", run_index=0, seed=seed, overrides={"data": {"batch_size": 256}})
print(f"bundle {time.time() - t:.1f}s", flush=True)
seed_everything(seed, torch.device("cpu"))
view = bundle.homogeneous(feats_type=0)
features = [f.coalesce() if f.is_sparse else f for f in view.features_list]
restore_legacy_message_functions()
source, destination = view.edge_index_without_self_loops
graph = dgl.graph((source.cpu(), destination.cpu()), num_nodes=view.num_nodes)
nodes = bundle.data.target_indices.cpu()
print(f"nodes={view.num_nodes} edges={graph.num_edges()} targets={nodes.numel()}", flush=True)
state_hash = hashlib.sha256(pickle.dumps(random.getstate(), protocol=4)).hexdigest()[:16]
key = hinormer.sequence_cache_key(bundle.dataset, graph, nodes, 50)
print(f"random_state_at_call={state_hash} cache_key={key[:16]}", flush=True)
t = time.time()
node_seq, status = hinormer.cached_node_sequences(graph, nodes, 50, dataset=bundle.dataset, cache_dir=cache_dir)
elapsed = time.time() - t
after = hashlib.sha256(pickle.dumps(random.getstate(), protocol=4)).hexdigest()[:16]
digest = hashlib.sha256(node_seq.numpy().tobytes()).hexdigest()[:16]
print(f"status={status} seconds={elapsed:.1f} shape={tuple(node_seq.shape)} node_seq_sha={digest} "
      f"random_state_after={after} first_random_after={random.random()!r}", flush=True)
print(f"maxrss_GB={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f}")
