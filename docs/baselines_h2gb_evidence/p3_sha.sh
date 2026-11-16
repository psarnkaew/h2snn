#!/bin/bash
# Card 1c: print the first 16 hex of each file's sha256 as this node sees it (NFS visibility check before a test run).
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
for f in "$@"; do printf '%s %s\n' "$(sha256sum "$f" | cut -c1-16)" "$f"; done
