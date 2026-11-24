import json
EV='/mnt/data/peeradon/h2snn-card19/docs/partM3b_evidence/'
GiB=2**30
def load(k):
    if k=='pdns':
        d=json.load(open(EV+'b5_pdns_check.json'))['loaded']
    else:
        d=json.load(open(EV+f'b2_counts_{k}.json'))[k]['loaded']
    nodes=d['nodes']; edges={tuple(e.split('|')):c for e,c in d['edges'].items()}
    fd={t:(v[0] if v else None) for t,v in d['feature_dims'].items()}
    return nodes,edges,fd
TARGET={'ogbn-mag':'paper','oag-cs':'paper','oag-eng':'paper','ieee-cis-g':'transaction','h-pokec':'user','pdns':'domain_node'}
def decoupled(k, width_featless, random_types=()):
    nodes,edges,fd=load(k); tgt=TARGET[k]
    pairs=[]
    for s,r,d in edges:
        if (s,d) not in pairs: pairs.append((s,d))
    for s,d in list(pairs):
        if s!=d and (d,s) not in pairs: pairs.append((d,s))
    w={t:(width_featless if (fd[t] is None or t in random_types) else fd[t]) for t in nodes}
    raw=sum(nodes[t]*w[t] for t in nodes)*4
    # hop1: key on d from s raw: width w[s]
    hop1={t:[] for t in nodes}
    for s,d in pairs: hop1[d].append(w[s])
    hop1_bytes=sum(nodes[t]*sum(ws) for t,ws in hop1.items())*4
    # hop2 only to target: for pairs (s,tgt), every hop1 key on s
    hop2=sum(sum(hop1[s]) for s,d in pairs if d==tgt)*nodes[tgt]*4
    tgt_store=(w[tgt]+sum(hop1[tgt]))*nodes[tgt]*4+hop2
    nkeys=1+len(hop1[tgt])+sum(len(hop1[s]) for s,d in pairs if d==tgt)
    return dict(raw=raw/GiB,hop1=hop1_bytes/GiB,hop2=hop2/GiB,peak=(raw+hop1_bytes+hop2)/GiB,target_store=tgt_store/GiB,nkeys=nkeys,
                row_width=int(tgt_store/4/nodes[tgt]))
for k in ['ogbn-mag','oag-cs','oag-eng','ieee-cis-g','pdns','h-pokec']:
    rt=('author','field_of_study','institution') if k=='ogbn-mag' else ()
    print('SeHGNN emb256', k, {a:(round(b,1) if isinstance(b,float) else b) for a,b in decoupled(k,256,rt).items()})
for k in ['oag-cs','oag-eng','ieee-cis-g','pdns','h-pokec']:
    print('PSHGCN generic', k, {a:(round(b,1) if isinstance(b,float) else b) for a,b in decoupled(k,256).items()})
# ogbn-mag PSHGCN their mag: P 128, others random 256
print('PSHGCN mag', {a:(round(b,1) if isinstance(b,float) else b) for a,b in decoupled('ogbn-mag',256,('author','field_of_study','institution')).items()})
print('mag label onehot GiB', 736389*349*4/GiB)
# freebase eye
fb={'book': 40402, 'film': 19427, 'music': 82351, 'sports': 1025, 'people': 17641, 'location': 9368, 'organization': 2731, 'business': 7153}
eye=sum(v*v for v in fb.values())*4
print('freebase eye total GiB', eye/GiB, 'music', 82351**2*4/GiB)
# hetero2net eye for H2GB featureless
for k in ['ieee-cis-g','h-pokec']:
    nodes,edges,fd=load(k)
    print('h2n eye',k, sum(nodes[t]**2 for t in nodes if fd[t] is None)*4/GiB)
# totals
for k in ['ogbn-mag','oag-cs','oag-eng','ieee-cis-g','pdns','h-pokec']:
    nodes,edges,fd=load(k)
    N=sum(nodes.values()); E=sum(edges.values())
    xb=sum(nodes[t]*(fd[t] or 0) for t in nodes)*4
    print(k,'N',N,'E',E,'X GiB',round(xb/GiB,2),'N*64*4 GiB',round(N*64*4/GiB,3))
