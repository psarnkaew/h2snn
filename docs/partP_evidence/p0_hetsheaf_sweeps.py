import yaml, glob, os, sys, json
root = sys.argv[1]
fams = ["diag_sheaf","bundle_sheaf","general_sheaf","attention_sheaf","diag_poly_sheaf","bundle_poly_sheaf","general_poly_sheaf"]
def fmt(v):
    if 'value' in v: return f"={v['value']}"
    if 'values' in v: return f"{v.get('distribution','')}{v['values']}"
    return f"{v.get('distribution')}[{v.get('min')},{v.get('max')}]"
for task in ["nc","lp","gc"]:
    for fam in fams:
        groups = {}
        for f in sorted(glob.glob(f"{root}/sweeps/{task}/*/{fam}.yaml")):
            y = yaml.safe_load(open(f))
            p = y['parameters']
            sig = {k: fmt(v) for k,v in p.items() if k!='dataset'}
            sig['_program'] = y['program']; sig['_method']=y.get('method'); sig['_run_cap']=y.get('run_cap'); sig['_metric']=str(y.get('metric')); sig['_early']=str(y.get('early_terminate'))
            key = json.dumps(sig, sort_keys=True)
            groups.setdefault(key, []).append(os.path.basename(os.path.dirname(f)))
        print(f"##### {task}/{fam}: {len(groups)} distinct")
        base=None
        for key, ds in groups.items():
            s=json.loads(key)
            if base is None:
                base=s; print("  datasets:", ds)
                for k,v in s.items(): print("   ",k,":",v)
            else:
                print("  datasets:", ds, "DIFF vs first:")
                for k in sorted(set(s)|set(base)):
                    if s.get(k)!=base.get(k): print("    ",k,":",base.get(k),"->",s.get(k))
