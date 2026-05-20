"""One-shot diagnostic — checks if preference_pairs.jsonl has structural pathologies that would make DPO collapse."""

import json
from collections import Counter
from pathlib import Path
from statistics import mean, median, stdev

PAIRS = Path(__file__).parent / "preference_pairs.jsonl"

pairs = []
with open(PAIRS, encoding="utf-8") as f:
    for line in f:
        pairs.append(json.loads(line))

print(f"=== {len(pairs)} pairs ===\n")

# --- 1. Source breakdown ---
print("--- by source (which version-pair generated each side) ---")
src_counter = Counter(p["source"] for p in pairs)
for src, n in src_counter.most_common():
    print(f"  {src:50s}  {n:5d}")

# --- 2. Tier breakdown ---
print("\n--- by tier ---")
tier_counter = Counter(p.get("tier", "normal") for p in pairs)
for tier, n in tier_counter.most_common():
    print(f"  {tier:10s}  {n:5d}")

# --- 3. Category breakdown ---
print("\n--- by category ---")
cat_counter = Counter(p["category"] for p in pairs)
for cat, n in cat_counter.most_common():
    print(f"  {cat:15s}  {n:5d}")

# --- 4. Length analysis (chars + word counts) ---
def text(msg_list):
    return msg_list[-1]["content"]

chosen_lens = [len(text(p["chosen"])) for p in pairs]
rejected_lens = [len(text(p["rejected"])) for p in pairs]

print("\n--- length (chars) ---")
print(f"  chosen   mean={mean(chosen_lens):6.1f}  median={median(chosen_lens):6.1f}  stdev={stdev(chosen_lens):6.1f}")
print(f"  rejected mean={mean(rejected_lens):6.1f}  median={median(rejected_lens):6.1f}  stdev={stdev(rejected_lens):6.1f}")

ratio = mean(chosen_lens) / mean(rejected_lens)
print(f"  chosen/rejected length ratio = {ratio:.3f}")

# Is chosen systematically longer or shorter than rejected (paired test)?
diffs = [c - r for c, r in zip(chosen_lens, rejected_lens)]
chosen_longer = sum(1 for d in diffs if d > 0)
rejected_longer = sum(1 for d in diffs if d < 0)
equal = sum(1 for d in diffs if d == 0)
print(f"  chosen longer: {chosen_longer} ({chosen_longer/len(pairs)*100:.1f}%)")
print(f"  rejected longer: {rejected_longer} ({rejected_longer/len(pairs)*100:.1f}%)")
print(f"  equal: {equal}")
print(f"  mean(chosen-rejected): {mean(diffs):+.1f} chars")

# Bucket the diffs to see distribution
buckets = [(-10000, -100), (-100, -50), (-50, -10), (-10, 10), (10, 50), (50, 100), (100, 10000)]
print("  diff distribution:")
for lo, hi in buckets:
    n = sum(1 for d in diffs if lo <= d < hi)
    bar = "#" * (n * 50 // len(pairs))
    print(f"    [{lo:+5d}, {hi:+5d}): {n:4d}  {bar}")

# --- 5. Per-source length analysis (catches "chosen always from newer model" effect) ---
print("\n--- length ratio per source ---")
for src in src_counter:
    src_pairs = [p for p in pairs if p["source"] == src]
    if not src_pairs:
        continue
    cl = [len(text(p["chosen"])) for p in src_pairs]
    rl = [len(text(p["rejected"])) for p in src_pairs]
    print(f"  {src:50s}  chosen={mean(cl):6.1f}  rejected={mean(rl):6.1f}  ratio={mean(cl)/mean(rl):.2f}  (n={len(src_pairs)})")

# --- 6. Chosen response uniqueness (catches degenerate "always pick same response") ---
chosen_set = set(text(p["chosen"]) for p in pairs)
rejected_set = set(text(p["rejected"]) for p in pairs)
print(f"\n--- response uniqueness ---")
print(f"  unique chosen responses:   {len(chosen_set):5d} / {len(pairs)}  ({len(chosen_set)/len(pairs)*100:.1f}%)")
print(f"  unique rejected responses: {len(rejected_set):5d} / {len(pairs)}  ({len(rejected_set)/len(pairs)*100:.1f}%)")

# --- 7. Empty / very short / very long responses ---
print(f"\n--- response length pathologies ---")
for label, lens in [("chosen", chosen_lens), ("rejected", rejected_lens)]:
    very_short = sum(1 for l in lens if l < 10)
    short = sum(1 for l in lens if 10 <= l < 30)
    very_long = sum(1 for l in lens if l > 500)
    print(f"  {label}: <10 chars: {very_short},  10-30 chars: {short},  >500 chars: {very_long}")

# --- 8. Token estimate (rough: 1 token ~ 3 chars for Russian) ---
print(f"\n--- estimated token counts (chars / 3) ---")
print(f"  chosen   mean={mean(chosen_lens)/3:5.1f}  max={max(chosen_lens)/3:5.1f}")
print(f"  rejected mean={mean(rejected_lens)/3:5.1f}  max={max(rejected_lens)/3:5.1f}")
