import hashlib
import json

import pytest

from evaluation.bbqv_loader import load_bbqv_manifest
from evaluation.bbqv_splits import (
    HammingTree, assign_groups, audit_partitions, family_id, group_records,
)


def row(i, family=None, pixel=None, dhash=None, category="Age"):
    identifier = f"01_{family if family is not None else i:02d}_{i:04d}_1_01"
    return {"example_id": identifier, "question_family": family_id(identifier),
            "pixel_sha256": pixel or f"pixel-{i}", "category": category,
            "dhash128": dhash or hashlib.md5(str(i).encode()).hexdigest()}


def test_question_family_preserves_image_and_polarity_variants():
    assert family_id("01_02_0001_1_01") == family_id("01_02_0099_2_04") == "01_02"
    with pytest.raises(ValueError):
        family_id("invalid-id")


def test_bk_tree_matches_exhaustive_hamming_search():
    values = [0, 1, 3, 7, 15, 255, 65535, (1 << 127)]
    tree = HammingTree()
    for i, value in enumerate(values):
        tree.add(value, i)
    for value in values + [63]:
        for radius in range(5):
            assert set(tree.query(value, radius)) == {
                i for i, candidate in enumerate(values) if (candidate ^ value).bit_count() <= radius}


def test_transitive_family_exact_pixel_and_near_duplicate_links():
    records = [row(1, family=1), row(2, family=1, pixel="shared"),
               row(3, pixel="shared", dhash="0" * 32),
               row(4, dhash="0" * 31 + "1"), row(5)]
    groups = group_records(records, near_duplicate_distance=1)
    assert sorted(map(len, groups.values())) == [1, 4]
    assert set(groups) == set(group_records(list(reversed(records)), 1))


def test_group_assignment_is_deterministic_and_exposed_families_stay_in_train():
    records = [row(i) for i in range(1, 31)] + [row(31, family=1)]
    groups = group_records(records)
    exposed = {records[0]["example_id"]}
    splits = assign_groups(groups, exposed)
    assert splits == assign_groups(dict(reversed(list(groups.items()))), exposed)
    assert audit_partitions(splits, exposed)["passed"]
    assert sum(map(len, splits.values())) == len(records)
    assert {r["example_id"] for r in splits["train"]} >= {
        records[0]["example_id"], records[-1]["example_id"]}
    leaked = {**splits, "test": splits["test"] + [splits["train"][0]]}
    assert not audit_partitions(leaked, exposed)["passed"]


def test_impossible_disjoint_split_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        assign_groups(group_records([row(1)]), set())


def test_audit_detects_near_duplicate_images_despite_different_group_labels():
    partitions = {
        "train": [{**row(1, dhash="0" * 32), "leakage_group": "one"}],
        "validation": [{**row(2, dhash="0" * 31 + "1"), "leakage_group": "two"}],
        "test": [{**row(3, dhash="f" * 32), "leakage_group": "three"}],
    }
    result = audit_partitions(partitions, set())
    assert not result["passed"]
    assert result["detected_cross_split_near_duplicate_edges"] == 1
    assert all(n == 0 for pair in result["cross_split_overlaps"].values() for n in pair.values())


@pytest.mark.parametrize("allow", [None, False, "true", True])
def test_shared_images_require_explicit_manifest_opt_in(tmp_path, allow):
    manifest = {"entries": [dict(example_id=str(i), image_path="same.jpg", category="Age",
                               source="ucf-crcv/BBQ-V", split="train", question="Q") for i in range(2)]}
    if allow is not None:
        manifest["allow_shared_images"] = allow
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    if allow is True:
        assert len(load_bbqv_manifest(path, require_images=False).examples) == 2
    else:
        with pytest.raises(ValueError):
            load_bbqv_manifest(path, require_images=False)
