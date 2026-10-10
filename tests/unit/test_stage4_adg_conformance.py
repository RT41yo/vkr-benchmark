from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from stage4_adg_conformance import explicit_q, paper_near, repository_near, tvd


def test_adg_near_witness_separates_paper_and_repository_semantics() -> None:
    values = [0.18, 0.11]
    target = 0.16
    assert paper_near(values, target) == 0
    assert repository_near(values, target) == 1


def test_adg_repository_semantics_can_change_full_induced_distribution() -> None:
    probabilities = (0.24, 0.18, 0.16, 0.11, 0.10, 0.08, 0.07, 0.06)
    paper_q = explicit_q(probabilities, repository_semantics=False)
    repo_q = explicit_q(probabilities, repository_semantics=True)
    assert np.isclose(paper_q.sum(), 1.0)
    assert np.isclose(repo_q.sum(), 1.0)
    assert tvd(paper_q, repo_q) > 0.0
