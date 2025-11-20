import re
from collections import defaultdict

LOGFILE = "debug_fly_groups.log"

# Patterns for group headers and entries
header_re = re.compile(r"\[(.*?)\]\s+(\d+)\s+flies")
entry_re = re.compile(r"idx=(\d+),")

groups = defaultdict(list)
current_group = None

with open(LOGFILE) as f:
    for line in f:
        h = header_re.search(line)
        if h:
            current_group = h.group(1)
            continue

        if current_group:
            e = entry_re.search(line)
            if e:
                idx = int(e.group(1))
                groups[current_group].append(idx)

# Convert lists → sets
group_sets = {k: set(v) for k, v in groups.items()}

# === Non-overlap summaries ===


def print_nonoverlap(setA, setB, nameA, nameB):
    inter = setA & setB
    onlyA = setA - setB
    onlyB = setB - setA

    print(f"\n=== {nameA} vs {nameB} ===")
    print(f"Intersection ({len(inter)}): {sorted(inter)}")
    print(f"{nameA} only ({len(onlyA)}): {sorted(onlyA)}")
    print(f"{nameB} only ({len(onlyB)}): {sorted(onlyB)}")


# ---- Pretty summary ----
all_group_names = list(group_sets.keys())

print("=== Groups loaded ===")
for g in all_group_names:
    print(f"{g:22s} : {len(group_sets[g])} entries")

print("\n=== Pairwise overlaps ===")
for g1 in all_group_names:
    for g2 in all_group_names:
        if g1 == g2:
            continue
        overlap = group_sets[g1] & group_sets[g2]
        print(f"{g1} ∩ {g2}: {len(overlap)}")

print("\n=== Detailed overlaps (idx lists) ===")
for g1 in all_group_names:
    for g2 in all_group_names:
        if g1 == g2:
            continue
        overlap = sorted(group_sets[g1] & group_sets[g2])
        print(f"\n{g1} ∩ {g2} ({len(overlap)} flies):")
        print(overlap)

# 1) SLI_TOP_LEARNERS vs STRONG_LEARNERS
print_nonoverlap(
    group_sets["SLI_TOP_LEARNERS"],
    group_sets["STRONG_LEARNERS"],
    "SLI_TOP_LEARNERS",
    "STRONG_LEARNERS",
)

# 2) SLI_TOP_LEARNERS vs FAST_LEARNERS
print_nonoverlap(
    group_sets["SLI_TOP_LEARNERS"],
    group_sets["FAST_LEARNERS"],
    "SLI_TOP_LEARNERS",
    "FAST_LEARNERS",
)
