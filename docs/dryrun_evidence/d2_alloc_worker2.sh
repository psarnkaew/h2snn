#!/bin/bash
# Card 2 D2, second worker in the author's allocation: small-cell lines 60-75, then the 14 H2GB cells (host memory is the
# allocation's 32G; a host OOM is recorded and retried on g3an01).
set -u
cd /mnt/data/peeradon/h2snn-dryrun/H2S-NN
E=../docs/dryrun_evidence
$E/d2_alloc_worker.sh $E/d2_cells_small.txt 60 75
$E/d2_alloc_worker.sh $E/d2_cells_h2gb.txt 1 14
