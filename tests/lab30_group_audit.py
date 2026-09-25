#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter

DATA_FILE = Path("results/lab30_trajectory_battery.jsonl")

records = []
with DATA_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

print("=" * 78)
print("LAB30 — AUDITORÍA DE ESTRUCTURA DE GRUPOS")
print("=" * 78)

print(f"\nN records: {len(records)}")

# 1. Campos disponibles
print("\n1. CAMPOS TOP-LEVEL")
print("-" * 78)

keys = sorted(set().union(*(r.keys() for r in records)))

for k in keys:
    vals = [r.get(k) for r in records]
    types = sorted(set(type(v).__name__ for v in vals))
    nonnull = sum(v is not None for v in vals)

    print(
        f"{k:<24} "
        f"present={nonnull}/{len(records)} "
        f"types={','.join(types)}"
    )

# 2. Campos candidatos a identificadores
print("\n2. CANDIDATOS A IDENTIFICADORES")
print("-" * 78)

candidate_names = [
    "id", "run_id", "rollout_id",
    "prompt_id", "problem_id", "task_id",
    "base_id", "example_id",
    "variant", "variant_id",
    "cycle", "cycle_id",
    "seed", "temperature",
    "family", "category"
]

for name in candidate_names:
    if name not in keys:
        continue

    vals = [r.get(name) for r in records]
    c = Counter(map(str, vals))

    print(f"\n{name}:")
    print(f"  unique = {len(c)}")
    print(f"  top    = {c.most_common(12)}")

# 3. Primer registro SIN trajectory
print("\n3. PRIMER REGISTRO — METADATA SOLAMENTE")
print("-" * 78)

r0 = dict(records[0])
r0.pop("trajectory", None)

print(json.dumps(r0, ensure_ascii=False, indent=2))

# 4. Distribución family × correct
print("\n4. MATRIZ FAMILY × CORRECT")
print("-" * 78)

families = sorted(set(str(r.get("family")) for r in records))

for fam in families:
    sub = [r for r in records if str(r.get("family")) == fam]
    correct = sum(bool(r.get("correct")) for r in sub)
    error = len(sub) - correct

    print(
        f"{fam:<18} "
        f"N={len(sub):<4} "
        f"correct={correct:<4} "
        f"error={error:<4}"
    )

# 5. Repetición de metadata
print("\n5. REPETICIÓN DE METADATA")
print("-" * 78)

meta_keys = [k for k in keys if k != "trajectory"]
signatures = []

for r in records:
    sig = tuple((k, str(r.get(k))) for k in meta_keys)
    signatures.append(sig)

counts = Counter(signatures)
repeated = [(n, c) for n, c in counts.items() if c > 1]

print(f"Unique metadata signatures : {len(counts)}")
print(f"Repeated signatures        : {len(repeated)}")

if repeated:
    print("\nTop repeated signatures:")
    for sig, count in sorted(repeated, key=lambda x: -x[1])[:20]:
        print(f"  repetitions={count}")
        print(f"  {dict(sig)}")

# 6. Trayectorias: longitud por familia y resultado
print("\n6. LONGITUD POR FAMILY × RESULTADO")
print("-" * 78)

for fam in families:
    sub = [r for r in records if str(r.get("family")) == fam]
    corr_lengths = [len(r["trajectory"]) for r in sub if r.get("correct")]
    err_lengths = [len(r["trajectory"]) for r in sub if not r.get("correct")]

    def fmt(x):
        if not x: return "n/a"
        return f"N={len(x)} mean={sum(x)/len(x):.2f} min={min(x)} max={max(x)}"

    print(f"{fam:<18}")
    print(f"  correct : {fmt(corr_lengths)}")
    print(f"  error   : {fmt(err_lengths)}")

print("\n" + "=" * 78)
print("FIN")
print("=" * 78)
