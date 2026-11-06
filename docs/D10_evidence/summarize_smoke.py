import json, math
rows = [json.loads(l) for l in open("../docs/D10_evidence/smoke_output.txt") if l.startswith("{")]
epochs = [r for r in rows if "epoch" in r]
for r in epochs:
    if r["epoch"] in (0, 1, 4, 9, 14, 19):
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
for r in rows:
    if "final_lambda_max_steps" in r or "stop" in r:
        print(r)
print("epochs:", len(epochs), "| max eta*lambda_max:", round(max(r["eta_lambda_max"] for r in epochs), 4),
      "| all losses finite:", all(math.isfinite(r["loss"]) for r in epochs),
      "| loss first/last:", round(epochs[0]["loss"], 4), round(epochs[-1]["loss"], 4))
