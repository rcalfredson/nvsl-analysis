from src.analysis.trajectory_turn_helpers import add_circle_turn_fields


def runCircleContactAnalysisUsingOutside(trj, va, radius_mm, opts):
    """
    • Ensures outside-circle stats exist (computes them once per trajectory)
    • Adds turning fields for *this* radius if not present
    • Returns a single-radius boundary_event_stats tree identical in shape
      to the wall contact result.
    """
    # compute once
    if "circle" not in trj.boundary_event_stats:
        trj._calcOutsideCirclePeriods()  # heavy loop

    stats_per_radius = trj.boundary_event_stats["circle"]["ctr"]["ctr"]

    if radius_mm not in stats_per_radius:
        raise ValueError(f"Radius {radius_mm} mm not found in trajectory stats")

    radius_stats = stats_per_radius[radius_mm]
    add_circle_turn_fields(trj, va, radius_stats, opts)

    # ⚠️ Wrap that single dict in the expected 3-level hierarchy
    return {"circle": {"ctr": {"ctr": radius_stats}}}
