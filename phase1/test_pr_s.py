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


if __name__ == "__main__":
    print("Task #58 — sticky subject tracking sanity check")
    test_acquire_picks_largest()
    test_lock_prefers_nearby_smaller()
    test_lock_fallthrough_on_big_jump()
    test_returns_none_when_volume_empty()
    print("\nAll checks passed.")
