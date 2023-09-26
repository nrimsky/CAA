import torch as t

from helpers import (
    project_onto_orthogonal_complement,
    add_vector_after_position,
    find_last_subtensor_position,
    find_instruction_end_postion,
)


def test_project_onto_orthogonal_complement():
    tensor = t.tensor([1.0, 2.0, 3.0])
    onto = t.tensor([1.0, 0.0, 0.0])
    result = project_onto_orthogonal_complement(tensor, onto)
    expected = t.tensor([0.0, 2.0, 3.0])
    assert t.allclose(result, expected)

    tensor = t.tensor([0.0, 5.0, 0.0])
    onto = t.tensor([1.0, 0.0, 0.0])
    result = project_onto_orthogonal_complement(tensor, onto)
    expected = t.tensor([0.0, 5.0, 0.0])
    assert t.allclose(result, expected)


def test_add_vector_after_position():
    matrix = t.tensor([[1, 2], [3, 4], [5, 6]], dtype=t.float32)
    vector = t.tensor([1, 1], dtype=t.float32)
    position_ids = t.tensor([1, 2, 3])
    result = add_vector_after_position(
        matrix, vector, position_ids, after=1, do_projection=False
    )
    expected = t.tensor([[1, 2], [4, 5], [6, 7]], dtype=t.float32)
    assert t.allclose(result, expected)


def test_find_last_subtensor_position():
    tensor = t.tensor([1, 2, 3, 4, 5, 1, 2, 3])
    sub_tensor = t.tensor([1, 2, 3])
    result = find_last_subtensor_position(tensor, sub_tensor)
    assert result == 5

    # Test case where sub_tensor isn't in tensor
    sub_tensor = t.tensor([6, 7])
    result = find_last_subtensor_position(tensor, sub_tensor)
    assert result == -1


def test_find_instruction_end_position():
    tokens = t.tensor([1, 2, 3, 4, 5, 1, 2, 3, 1, 2, 3])
    end_str = t.tensor([1, 2, 3])
    result = find_instruction_end_postion(tokens, end_str)
    assert result == 10

    # Test case where end_str isn't in tokens
    end_str = t.tensor([6, 7])
    result = find_instruction_end_postion(tokens, end_str)
    assert result == -1