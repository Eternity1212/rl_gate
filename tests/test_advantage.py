import numpy as np

from triage_grpo.advantage import compute_group_advantages, group_normalize
from triage_grpo.metrics import hack_rate
from triage_grpo.types import Cell, GroupInput


def test_group_normalize_zero_var():
    out = group_normalize(np.array([1.0, 1.0, 1.0]))
    assert np.allclose(out, 0.0)


def test_papo_process_only_on_correct():
    g = GroupInput(
        outcome_rewards=[1, 1, 0, 0],
        process_scores=[0.9, 0.1, 0.95, 0.05],
        confidences=[0.5, 0.5, 0.5, 0.5],
    )
    out = compute_group_advantages(g, method="papo", alpha=1.0, tau=0.5)
    # Wrong answers must have zero process advantage
    assert out.a_process[2] == 0.0
    assert out.a_process[3] == 0.0
    # Among correct, higher process gets higher A_p
    assert out.a_process[0] > out.a_process[1]


def test_triage_filter_masks_w_high():
    g = GroupInput(
        outcome_rewards=[1, 1, 0, 0],
        process_scores=[0.9, 0.2, 0.1, 0.85],
        confidences=[0.8, 0.7, 0.3, 0.9],
    )
    out = compute_group_advantages(
        g, method="triage", tau=0.5, w_high_mode="filter", gamma=0.1
    )
    assert out.cells == [Cell.C_HIGH, Cell.C_LOW, Cell.W_LOW, Cell.W_HIGH]
    assert out.masks == [True, True, True, False]
    assert out.advantages[3] == 0.0
    assert hack_rate(out.cells) == 0.25


def test_triage_downweight_keeps_mask_true():
    g = GroupInput(
        outcome_rewards=[1, 0],
        process_scores=[0.9, 0.9],
        confidences=[0.5, 0.5],
    )
    out = compute_group_advantages(
        g, method="triage", tau=0.5, w_high_mode="downweight", beta=1.0, alpha=0.0, gamma=0.0
    )
    assert out.masks == [True, True]
    assert out.cells[1] == Cell.W_HIGH
    assert out.a_hack[1] < 0.0


def test_p_grpo_gates_process_on_wrong():
    g = GroupInput(
        outcome_rewards=[1, 0],
        process_scores=[0.5, 1.0],
    )
    out = compute_group_advantages(g, method="p_grpo")
    # gated rewards: [1+0.5, 0] = [1.5, 0] -> first higher
    assert out.advantages[0] > out.advantages[1]


def test_prof_filters_conflicts():
    g = GroupInput(
        outcome_rewards=[1, 1, 0, 0],
        process_scores=[0.9, 0.1, 0.1, 0.9],  # C_HIGH, C_LOW, W_LOW, W_HIGH
    )
    out = compute_group_advantages(g, method="prof", tau=0.5)
    assert out.masks == [True, False, True, False]


def test_hand_calc_orm():
    # r_o = [1,0], mean=0.5, std=0.5 -> A=[1,-1]
    g = GroupInput(outcome_rewards=[1, 0], process_scores=[0.0, 0.0])
    out = compute_group_advantages(g, method="orm_grpo")
    assert np.allclose(out.advantages, [1.0, -1.0], atol=1e-6)
