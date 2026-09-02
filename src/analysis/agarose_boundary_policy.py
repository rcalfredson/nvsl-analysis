"""Boundary-state policies for the dual-circle agarose metric."""

AGAROSE_BOUNDARY_POLICY_HYSTERETIC = "hysteretic"
AGAROSE_BOUNDARY_POLICY_LEGACY = "legacy"
AGAROSE_BOUNDARY_POLICIES = (
    AGAROSE_BOUNDARY_POLICY_HYSTERETIC,
    AGAROSE_BOUNDARY_POLICY_LEGACY,
)


def normalize_agarose_boundary_policy(value) -> str:
    """Return a validated dual-circle boundary policy name."""
    policy = str(value)
    if policy not in AGAROSE_BOUNDARY_POLICIES:
        choices = ", ".join(repr(choice) for choice in AGAROSE_BOUNDARY_POLICIES)
        raise ValueError(
            f"agarose dual-circle boundary policy must be one of {choices}; "
            f"got {policy!r}"
        )
    return policy
