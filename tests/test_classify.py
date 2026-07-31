from triage_grpo.classify import assign_cell, assign_cells
from triage_grpo.types import Cell


def test_assign_four_cells():
    assert assign_cell(1, 0.9, 0.5) == Cell.C_HIGH
    assert assign_cell(1, 0.1, 0.5) == Cell.C_LOW
    assert assign_cell(0, 0.1, 0.5) == Cell.W_LOW
    assert assign_cell(0, 0.9, 0.5) == Cell.W_HIGH


def test_assign_cells_batch():
    cells = assign_cells([1, 1, 0, 0], [0.9, 0.2, 0.1, 0.85], tau=0.5)
    assert cells == [Cell.C_HIGH, Cell.C_LOW, Cell.W_LOW, Cell.W_HIGH]
