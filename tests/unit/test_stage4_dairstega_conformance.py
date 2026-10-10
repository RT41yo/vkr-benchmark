from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from stage4_dairstega_conformance import (
    DAIRFixture,
    Interval,
    paper_allocate,
    prefix_length_paper,
    prefix_length_repository,
    repository_allocate,
)


def test_dairstega_linear_literal_allocation_matches_paper_power_rule() -> None:
    fixture = DAIRFixture(
        "linear",
        (0.44, 0.24, 0.14, 0.08, 0.05, 0.03, 0.02),
        8,
        1.0,
        100,
    )
    assert repository_allocate(fixture) == paper_allocate(fixture)


def test_dairstega_sqrt_rounding_can_change_repository_allocation() -> None:
    # This fixture is deliberately close enough to integer boundaries that the
    # pinned four-decimal sqrt rounding changes at least one roulette boundary.
    fixture = DAIRFixture(
        "sqrt-rounding",
        (
            0.000244453319355365,
            0.00034143519970818994,
            0.010050976875227857,
            0.0022990481763515247,
            0.007032330521554073,
            0.09975400611673797,
            0.12587087776967712,
            0.01090729913362127,
            0.1270194760506005,
            0.3804810894889217,
            0.025449991132712443,
            0.015665524108526507,
            0.007696732813895967,
            0.00020103521899825818,
            0.0002264716529964316,
            0.18675925242111469,
        ),
        12,
        0.5,
        16,
    )
    assert repository_allocate(fixture) != paper_allocate(fixture)


def test_dairstega_singleton_prefix_separates_paper_and_repository() -> None:
    interval = Interval(token_id=0, start=37, stop=38)
    assert prefix_length_paper(interval, 8) == 8
    assert prefix_length_repository(interval, 8) == 7
