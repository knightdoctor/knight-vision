"""
phase1/clustering.py
====================
DBSCAN-based clustering of residual point clouds.

After background subtraction, the residual point cloud is clustered to
isolate distinct moving objects.  The pipeline then selects the single
cluster most likely to be the monitored subject and passes its frame
history to the respiratory analyser.

Usage
-----
    from phase1.clustering import cluster_residuals, select_subject_cluster
    import numpy as np

    residuals = np.random.randn(200, 3)
    clusters  = cluster_residuals(residuals, config)
    bbox = np.array([[-1.5, -1.5, 0.0], [1.5, 1.5, 2.5]])
    subject   = select_subject_cluster(clusters, bbox)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.cluster import DBSCAN


@dataclass
class Cluster:
    """A single labelled cluster from the residual point cloud.

    Attributes
    ----------
    points : np.ndarray
        Shape (N, 3) — XYZ coordinates of points in this cluster.
    centroid : np.ndarray
        Shape (3,) — mean XYZ of the cluster.
    bbox_min : np.ndarray
        Shape (3,) — axis-aligned bounding box minimum corner.
    bbox_max : np.ndarray
        Shape (3,) — axis-aligned bounding box maximum corner.
    n_points : int
        Number of points.
    """
    points:   np.ndarray
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    n_points: int


def cluster_residuals(points: np.ndarray, config) -> List[Cluster]:
    """Cluster a residual point cloud using DBSCAN.

    Parameters
    ----------
    points : np.ndarray
        Shape (N, 3) residual points (output of BackgroundModel.subtract).
    config : KVConfig
        Pipeline configuration.  Reads ``dbscan_eps``,
        ``dbscan_min_samples``, ``cluster_min_points``.

    Returns
    -------
    list of Cluster
        Clusters sorted by point count **descending** (largest first).
        Clusters with fewer than ``config.cluster_min_points`` points are
        excluded.  DBSCAN noise points (label -1) are discarded.
        Returns an empty list if *points* is empty or no clusters survive
        the minimum-points filter.
    """
    if points.shape[0] < config.cluster_min_points:
        return []

    db = DBSCAN(
        eps=config.dbscan_eps,
        min_samples=config.dbscan_min_samples,
        algorithm="ball_tree",
        n_jobs=1,
    ).fit(points)

    labels = db.labels_
    unique_labels = set(labels) - {-1}   # exclude noise

    clusters: List[Cluster] = []
    for label in unique_labels:
        mask = labels == label
        cluster_pts = points[mask]
        n = cluster_pts.shape[0]
        if n < config.cluster_min_points:
            continue
        centroid = cluster_pts.mean(axis=0)
        bbox_min = cluster_pts.min(axis=0)
        bbox_max = cluster_pts.max(axis=0)
        clusters.append(Cluster(
            points=cluster_pts,
            centroid=centroid,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            n_points=n,
        ))

    # Sort largest first
    clusters.sort(key=lambda c: c.n_points, reverse=True)
    return clusters


def is_human_shaped(
    cluster: Cluster,
    y_min_m: float = 0.20,
    y_max_m: float = 1.80,
    xz_max_m: float = 1.00,
) -> bool:
    """Reject clusters whose bounding box is incompatible with a human.

    Wall-slice or sofa-edge residuals that get DBSCAN-clustered tend to
    be horizontally elongated and short in Y; humans always have at
    least ~20 cm of vertical extent (newborn lying) and at most ~1.8 m
    (adult standing). Returns True if the cluster's bbox fits a generous
    human envelope, False otherwise.

    Set ``y_min_m = 0.0`` to disable the gate (returns True always).
    """
    if y_min_m <= 0.0:
        return True
    span = cluster.bbox_max - cluster.bbox_min
    y_span = float(span[1])
    if not (y_min_m <= y_span <= y_max_m):
        return False
    xz_span = float(max(span[0], span[2]))
    if xz_span > xz_max_m:
        return False
    return True


def select_subject_cluster(
    clusters: List[Cluster],
    monitoring_volume_bbox: np.ndarray,
    prev_centroid: Optional[np.ndarray] = None,
    lock_radius_m: float = 0.30,
    size_dominance_ratio: float = 2.0,
    shape_y_min_m: float = 0.0,
    shape_y_max_m: float = 1.80,
    shape_xz_max_m: float = 1.00,
) -> Optional[np.ndarray]:
    """Select the most likely subject cluster from a list of clusters.

    Two-mode selection (task #58 — temporal coherence):

    1. **Locked mode** (``prev_centroid`` provided): pick the cluster
       whose centroid is closest to ``prev_centroid`` *and* within
       ``lock_radius_m``. Lock applies only to clusters whose centroid
       is inside the monitoring volume. If no in-volume cluster is
       within the lock radius, fall through to acquire mode.

    2. **Acquire mode** (no prev_centroid, or lock fell through): pick
       the **largest** cluster whose centroid is in the monitoring
       volume. This is the original behaviour.

    The split prevents per-frame flicker between the subject and a
    similarly-sized background cluster (e.g. sofa, wall edge) — once
    we've found the subject, we stick with whichever cluster is
    spatially closest until we lose it for long enough that the
    pipeline clears the lock.

    Parameters
    ----------
    clusters : list of Cluster
        Output of :func:`cluster_residuals` (already sorted by size).
    monitoring_volume_bbox : np.ndarray
        Shape (2, 3) — ``[[xmin, ymin, zmin], [xmax, ymax, zmax]]``
        in metres.
    prev_centroid : np.ndarray, optional
        Shape (3,) — last frame's subject centroid. When provided,
        enables locked mode.
    lock_radius_m : float
        Maximum centroid jump per frame that still counts as the same
        subject. 0.30 m at 5-10 fps = 1.5-3 m/s of permitted motion,
        which comfortably covers sitting fidget and slow rotation
        without bleeding into a background cluster ~1 m away.

    Returns
    -------
    np.ndarray or None
        Shape (N, 3) point cloud of the selected cluster, or ``None``
        if no cluster centroid falls within the monitoring volume.
    """
    if not clusters:
        return None

    bbox_min = monitoring_volume_bbox[0]
    bbox_max = monitoring_volume_bbox[1]

    in_volume = [
        c for c in clusters
        if np.all(c.centroid >= bbox_min) and np.all(c.centroid <= bbox_max)
        and is_human_shaped(c, shape_y_min_m, shape_y_max_m, shape_xz_max_m)
    ]
    if not in_volume:
        return None

    # Locked mode: prefer the cluster nearest the previous centroid,
    # provided it sits within lock_radius_m AND no much-larger cluster
    # also sits in the volume. The size-dominance check breaks a wedged
    # lock when the previous candidate was actually a small persistent
    # background residual (e.g. a 250-pt wall phantom) and the real
    # subject has since walked in as a 5000-pt cluster well outside the
    # lock radius: without the override, the lock chases the phantom
    # forever because the phantom centroid is closest to itself.
    if prev_centroid is not None:
        prev = np.asarray(prev_centroid, dtype=float)
        best = min(in_volume, key=lambda c: np.linalg.norm(c.centroid - prev))
        if np.linalg.norm(best.centroid - prev) <= lock_radius_m:
            # in_volume[0] is the largest by point count (cluster_residuals
            # sorts largest-first). Take it if it dominates the locked
            # candidate by size_dominance_ratio; otherwise hold the lock.
            largest = in_volume[0]
            if (largest is not best and
                    largest.n_points >= size_dominance_ratio * best.n_points):
                return largest.points
            return best.points
        # Fall through to acquire mode if no candidate is within radius.

    # Acquire mode: largest in-volume cluster (clusters are sorted
    # largest-first by cluster_residuals).
    return in_volume[0].points


def select_chest_subset(
    subject_points: np.ndarray,
    y_band_frac: tuple = (0.50, 0.85),
    min_points: int = 30,
    xz_radius_m: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Pick the upper ~chest portion of a subject point cloud by Y, then
    crop laterally around the cluster's median X/Z centre.

    Subject is assumed to be sitting/standing in the shared frame
    (X right, **Y up**, Z forward). We keep points whose Y is in
    ``[y_min + lo*range, y_min + hi*range]`` of the cluster's Y span.
    The lower bound excludes legs/abdomen (which translate with body
    sway but don't expand with breath); the upper bound excludes the
    head (so head bobs don't dominate).

    PR T (2026-05-18): when ``xz_radius_m`` is set, also drop points
    whose X or Z deviates from the cluster's median X/Z by more than
    ``xz_radius_m`` metres. This stops the analysis subset from bleeding
    onto adjacent furniture/wall residuals that DBSCAN chains into the
    same cluster when the subject's arm/desk/chair-back narrows the gap
    below ``dbscan_eps``. Median is used (not mean) so the centre stays
    anchored on the subject's torso even when a tail of cluster points
    leaks toward background structure.

    Returns ``None`` if the subject is too small or the chest band is
    too sparse — in that case the caller should fall back to the
    whole-subject centroid.
    """
    if subject_points is None or subject_points.shape[0] == 0:
        return None
    y = subject_points[:, 1]
    y_range = float(y.max() - y.min())
    if y_range < 0.20:                  # cluster too short to call "body"
        return None
    band_lo = float(y.min()) + y_band_frac[0] * y_range
    band_hi = float(y.min()) + y_band_frac[1] * y_range
    chest = subject_points[(y >= band_lo) & (y <= band_hi)]
    if xz_radius_m is not None and chest.shape[0] > 0:
        # Anchor on the FULL cluster's median X/Z (not the Y-band subset's).
        # If the cluster bleeds into furniture, the Y-band slice can end up
        # dominated by furniture points — anchoring on its own median would
        # then lock the crop onto the furniture. The full-cluster median is
        # weighted toward whichever sub-region is densest, which for a real
        # subject cluster is the torso blob, not the leaked-into-background
        # tail.
        x_anchor = float(np.median(subject_points[:, 0]))
        z_anchor = float(np.median(subject_points[:, 2]))
        in_box = ((np.abs(chest[:, 0] - x_anchor) <= xz_radius_m) &
                  (np.abs(chest[:, 2] - z_anchor) <= xz_radius_m))
        chest = chest[in_box]
    if chest.shape[0] < min_points:
        return None
    return chest
