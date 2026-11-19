"""Grouped project partitions of the official BBQ-V evaluation corpus.

No training is performed. Related question families, exact images and detected
near-duplicate images are kept in one partition, including transitive links.
"""

import hashlib
import json
import re
from collections import Counter, defaultdict


class UnionFind:
    def __init__(self, n):
        self.parents = list(range(n))

    def find(self, i):
        while self.parents[i] != i:
            self.parents[i] = self.parents[self.parents[i]]
            i = self.parents[i]
        return i

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parents[max(a, b)] = min(a, b)


class HammingTree:
    """BK-tree for bounded Hamming search on 128-bit perceptual hashes."""

    def __init__(self):
        self.root = None

    def add(self, value, index):
        if self.root is None:
            self.root = (value, index, {})
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                return
            if distance not in node[2]:
                node[2][distance] = (value, index, {})
                return
            node = node[2][distance]

    def query(self, value, radius):
        stack = [self.root] if self.root else []
        matches = []
        while stack:
            current, index, children = stack.pop()
            distance = (current ^ value).bit_count()
            if distance <= radius:
                matches.append(index)
            stack.extend(child for edge, child in children.items()
                         if distance - radius <= edge <= distance + radius)
        return matches


def family_id(example_id):
    if not re.fullmatch(r"\d+_\d+_\d+_[12]_\d+", example_id):
        raise ValueError(f"Unexpected official BBQ-V id: {example_id!r}")
    # Keep all scenarios, images, and polarity variants for a question family
    # together, rather than splitting the last image-number token randomly.
    return "_".join(example_id.split("_")[:2])


def group_records(records, near_duplicate_distance=4):
    if not 0 <= near_duplicate_distance <= 8:
        raise ValueError("near_duplicate_distance must be in [0,8]")
    ids = [r["example_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate example ids in source")
    union = UnionFind(len(records))
    seen = {}
    hashes = HammingTree()
    for i, row in enumerate(records):
        for kind, value in (("family", family_id(row["example_id"])),
                            ("pixels", row["pixel_sha256"])):
            key = (kind, value)
            if key in seen:
                union.union(i, seen[key])
            else:
                seen[key] = i
        value = int(row["dhash128"], 16)
        for prior in hashes.query(value, near_duplicate_distance):
            union.union(i, prior)
        hashes.add(value, i)
    groups = defaultdict(list)
    for i, row in enumerate(records):
        groups[union.find(i)].append(row)
    # Stable component id regardless of source physical order.
    result = {}
    for rows in groups.values():
        key = hashlib.sha256("\n".join(sorted(r["example_id"] for r in rows)).encode()).hexdigest()[:20]
        result[key] = sorted(rows, key=lambda r: r["example_id"])
    return result


def assign_groups(groups, exposed_ids, seed=42, ratios=(0.7, 0.15, 0.15)):
    if len(ratios) != 3 or any(r <= 0 for r in ratios) or abs(sum(ratios) - 1) > 1e-9:
        raise ValueError("Three positive split ratios must sum to one")
    names = ("train", "validation", "test")
    totals = Counter(row["category"] for rows in groups.values() for row in rows)
    actual = {s: Counter() for s in names}
    assignments = {}
    counts = {gid: Counter(r["category"] for r in rows) for gid, rows in groups.items()}
    for gid, rows in groups.items():
        if any(r["example_id"] in exposed_ids for r in rows):
            assignments[gid] = "train"
            actual["train"].update(counts[gid])
    order = sorted((g for g in groups if g not in assignments), key=lambda gid: (
        -len(groups[gid]), hashlib.sha256(f"{seed}:{gid}".encode()).hexdigest()))
    for gid in order:
        def cost(split):
            ratio = ratios[names.index(split)]
            # Delta of a category-normalised quadratic balance objective.
            return sum(((actual[split][cat] + number - ratio * totals[cat]) ** 2
                        - (actual[split][cat] - ratio * totals[cat]) ** 2) / max(1, totals[cat])
                       for cat, number in counts[gid].items())
        choice = min(names, key=lambda split: (cost(split), names.index(split)))
        assignments[gid] = choice
        actual[choice].update(counts[gid])
    # If only one unexposed component remains for a category, prioritise the
    # final test over validation. Never move an exposed component out of train.
    for category in sorted(totals):
        if actual["test"][category]:
            continue
        candidates = [gid for gid in order if assignments[gid] == "validation"
                      and counts[gid][category] > 0
                      and sum(assignments[g] == "validation" for g in order) > 1]
        if candidates:
            gid = min(candidates, key=lambda g: (len(groups[g]), g))
            assignments[gid] = "test"
            actual["validation"].subtract(counts[gid])
            actual["test"].update(counts[gid])
    partitions = {name: [] for name in names}
    for gid, rows in groups.items():
        partitions[assignments[gid]].extend({**row, "leakage_group": gid} for row in rows)
    for name in names:
        partitions[name].sort(key=lambda r: r["example_id"])
        if not partitions[name]:
            raise ValueError(f"Grouping left an empty {name} split; do not relax leakage checks silently")
    return partitions


def audit_partitions(partitions, exposed_ids, near_duplicate_distance=4):
    overlaps = {}
    names = list(partitions)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            values = {}
            for key in ("example_id", "pixel_sha256", "leakage_group", "question_family"):
                shared = {r[key] for r in partitions[a]} & {r[key] for r in partitions[b]}
                values[key] = len(shared)
            overlaps[a + "__" + b] = values
    exposed_outside_train = sorted({r["example_id"] for name in ("validation", "test")
                                   for r in partitions[name]} & set(exposed_ids))
    # Recompute image-neighbour edges independently of stored group labels.
    tree = HammingTree()
    hash_owners = {}
    hash_rows = []
    near_cross_split = 0
    for split, rows in partitions.items():
        for row in rows:
            value = int(row["dhash128"], 16)
            if (split, value) in hash_owners:
                continue
            for previous in tree.query(value, near_duplicate_distance):
                if hash_rows[previous] != split:
                    near_cross_split += 1
            tree.add(value, len(hash_rows))
            hash_rows.append(split)
            hash_owners[(split, value)] = True
    passed = (not exposed_outside_train and near_cross_split == 0
              and all(n == 0 for pair in overlaps.values() for n in pair.values()))
    return {"passed": passed, "cross_split_overlaps": overlaps,
            "detected_cross_split_near_duplicate_edges": near_cross_split,
            "previously_exposed_ids_outside_train": exposed_outside_train,
            "near_duplicate_note": "dHash edges were included in connected groups; this is a heuristic, not a guarantee against all semantic duplicates."}


def content_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
