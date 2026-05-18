"""Sanity check for task #58 — sticky subject tracking.

Builds synthetic Cluster lists with two candidates (one resembling the
subject at z~1.0 m, one resembling a background blob at z~1.5 m) and
checks that:

  1. Acquire mode (prev_centroid=None) returns the LARGEST cluster.
  2. Locked mode prefers the cluster closest to prev_centroid even when
     a larger candidate exists.
  3. Locked mode falls through to acquire when the closest candidate is
     farther than lock_radius_m.

Run with:  python3 phase1/test_pr_s.py
"""
import sys
sys.path.insert(0, ".")
import numpy as np
from phase1.clustering import Cluster, select_subject_cluster

VOLUME = np.array([[-1.5, -1.5, 0.5],
                   [ 1.5,  1.5, 2.0]], dtype=float)


def _make(centroid, n_points):
    """Build a degenerate Cluster object — only centroid + n_points matter
    for the selector. points/bbox just need to round-trip."""
    pts = np.tile(np.asarray(centroid, dtype=float), (n_points, 1))
    return Cluster(
        points=pts,
        centroid=np.asarray(centroid, dtype=float),
        bbox_min=pts.min(axis=0),
        bbox_max=pts.max(axis=0),
        n_points=n_points,
    )


def test_acquire_picks_largest():
    subject = _make([0.0, 0.0, 1.0], n_points=400)   # the person
    sofa    = _make([0.0, 0.0, 1.5], n_points=600)   # bigger background blob
    # cluster_residuals returns clusters sorted largest-first.
    clusters = [sofa, subject]
    out = select_subject_cluster(clusters, VOLUME, prev_centroid=None)
    assert out is sofa.points, "Acquire mode should pick LARGEST in volume"
    print("  [ok] acquire mode picks largest in volume")


def test_lock_prefers_nearby_smaller():
    subject = _make([0.0, 0.0, 1.0], n_points=400)
    sofa    = _make([0.0, 0.0, 1.5], n_points=600)
    clusters = [sofa, subject]
    # Previous frame had subject at [0,0,1.02] — within 0.30 m of subject,
    # but >0.30 m from sofa.
    out = select_subject_cluster(
        clusters, VOLUME,
        prev_centroid=np.array([0.0, 0.0, 1.02]),
        lock_radius_m=0.30,
    )
    assert out is subject.points, (
        "Locked mode should prefer subject (close to prev) over larger sofa"
    )
    print("  [ok] lock prefers nearby smaller over distant larger")


def test_lock_fallthrough_on_big_jump():
    subject = _make([0.0, 0.0, 1.0], n_points=400)
    sofa    = _make([0.0, 0.0, 1.5], n_points=600)
    clusters = [sofa, subject]
    # Previous centroid 2 m away from any cluster → outside lock radius.
    out = select_subject_cluster(
        clusters, VOLUME,
        prev_centroid=np.array([0.0, 0.0, 3.0]),   # behind the room
        lock_radius_m=0.30,
    )
    assert out is sofa.points, (
        "When no cluster is within lock_radius, should fall through to "
        "largest-in-volume (acquire mode)"
    )
    print("  [ok] lock falls through to acquire when prev far away")


def test_returns_none_when_volume_empty():
    far = _make([0.0, 0.0, 5.0], n_points=400)     # outside Z range
    out = select_subject_cluster([far], VOLUME, prev_centroid=None)
    assert out is None
    print("  [ok] returns None when no cluster in volume")


def test_size_dominance_breaks_phantom_lock():
    """Real bug seen 2026-05-18: lock latched onto a 250-pt persistent
    wall residual at z=1.9; subject walked in as a 5000-pt cluster at
    z=1.0 but the lock kept tracking the phantom because the phantom
    centroid was closest to itself. Dominance override should release."""
    phantom = _make([0.0, 0.0, 1.9], n_points=260)   # the wall residual
    subject = _make([0.0, 0.0, 1.0], n_points=5000)  # Phil walks in
    # cluster_residuals returns largest-first; subject is now largest.
    clusters = [subject, phantom]
    out = select_subject_cluster(
        clusters, VOLUME,
        prev_centroid=np.array([0.0, 0.0, 1.9]),   # was locked on phantom
        lock_radius_m=0.30,
        size_dominance_ratio=2.0,
    )
    assert out is subject.points, (
        "Dominance override should switch from 260-pt phantom to "
        "5000-pt real subject (>=2x larger)"
    )
    print("  [ok] dominance override breaks phantom lock")


def test_dominance_does_not_fire_for_similar_sizes():
    """The 1.5x sofa-vs-subject case from test_lock_prefers_nearby_smaller
    should still hold — sofa is 600/400 = 1.5x, below the 2.0 dominance
    ratio, so the lock should keep the nearby subject."""
    subject = _make([0.0, 0.0, 1.0], n_points=400)
    sofa    = _make([0.0, 0.0, 1.5], n_points=600)
    clusters = [sofa, subject]
    out = select_subject_cluster(
        clusters, VOLUME,
        prev_centroid=np.array([0.0, 0.0, 1.0]),
        lock_radius_m=0.30,
        size_dominance_ratio=2.0,
    )
    assert out is subject.points, (
        "1.5x size difference should not trigger dominance override"
    )
    print("  [ok] dominance leaves lock alone for similar-size clusters")


def _make_box(centroid, span_xyz, n_points=400):
    """Build a Cluster with a configurable bbox (used to test shape gate)."""
    c = np.asarray(centroid, dtype=float)
    s = np.asarray(span_xyz, dtype=float) / 2.0
    pts = np.random.uniform(c - s, c + s, size=(n_points, 3))
    return Cluster(
        points=pts,
        centroid=pts.mean(axis=0),
        bbox_min=pts.min(axis=0),
        bbox_max=pts.max(axis=0),
        n_points=n_points,
    )


def test_shape_gate_rejects_thin_wall_slice():
    """A wall-slice phantom has tall X span, tiny Y span — fails y_min."""
    np.random.seed(1)
    wall = _make_box([0.0, 1.0, 1.5], span_xyz=[0.8, 0.05, 0.02], n_points=600)
    subj = _make_box([0.0, 1.0, 1.0], span_xyz=[0.4, 0.8,  0.30], n_points=400)
    out = select_subject_cluster(
        [wall, subj], VOLUME, prev_centroid=None,
        shape_y_min_m=0.20, shape_y_max_m=1.80, shape_xz_max_m=1.00,
    )
    assert out is subj.points, (
        "Shape gate should reject the 5cm-tall wall slice and pick the "
        "human-shaped subject even though wall has more points"
    )
    print("  [ok] shape gate rejects thin wall slice")


def test_shape_gate_disabled_by_default():
    """shape_y_min_m=0.0 (default) bypasses the gate entirely."""
    np.random.seed(1)
    wall = _make_box([0.0, 1.0, 1.5], span_xyz=[0.8, 0.05, 0.02], n_points=600)
    out = select_subject_cluster([wall], VOLUME, prev_centroid=None)
    assert out is wall.points, "Default shape_y_min_m=0.0 should not gate"
    print("  [ok] shape gate disabled by default")


if __name__ == "__main__":
    print("Task #58 — sticky subject tracking sanity check")
    test_acquire_picks_largest()
    test_lock_prefers_nearby_smaller()
    test_lock_fallthrough_on_big_jump()
    test_returns_none_when_volume_empty()
    test_size_dominance_breaks_phantom_lock()
    test_dominance_does_not_fire_for_similar_sizes()
    test_shape_gate_rejects_thin_wall_slice()
    test_shape_gate_disabled_by_default()
    print("\nAll checks passed.")
