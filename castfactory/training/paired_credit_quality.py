"""Versioned numerical utility in raw full-horizon WAPE units.

Only v7 opts into the bounded map. Cost and contract terms remain separate;
this is component scaling, never within-group normalization or WAPE clipping.
"""
from castfactory.training.full_agent_metrics import metric_context, metric_decimal
from castfactory.training.paired_credit_objective import BOUNDED_NATURAL_OBJECTIVE_VERSION


def quality(wape, objective_version=None):
    value = metric_decimal(wape)
    if value < 0:
        raise ValueError("Quality requires nonnegative raw WAPE")
    with metric_context():
        if objective_version == BOUNDED_NATURAL_OBJECTIVE_VERSION:
            return -4 * value / (value + metric_decimal(".10"))
        return -value / metric_decimal('.10')


def quality_gain(before, after, objective_version=None):
    """Full-H after-minus-before quality, with exact historical arithmetic."""
    with metric_context():
        if objective_version == BOUNDED_NATURAL_OBJECTIVE_VERSION:
            return quality(after, objective_version) - quality(before, objective_version)
        return 10 * (metric_decimal(before) - metric_decimal(after))


def initial_allocation_scale(baseline_wape, retained_wape, objective_version=None):
    """Secant slope allocating one full-H improvement over accepted segments.

    Missing segments retain the public baseline; their error difference is
    zero. Never transform each segment's partial WAPE independently.
    """
    with metric_context():
        if objective_version == BOUNDED_NATURAL_OBJECTIVE_VERSION:
            baseline, retained = metric_decimal(baseline_wape), metric_decimal(retained_wape)
            if min(baseline, retained) < 0:
                raise ValueError("INITIAL allocation requires nonnegative errors")
            scale = metric_decimal(".10")
            return 4 * scale / ((baseline + scale) * (retained + scale))
        return metric_decimal(10)
