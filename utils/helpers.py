import torch as t

def project_onto_orthogonal_complement(tensor, onto):
    """
    Projects tensor onto the orthogonal complement of the span of onto.
    """
    # Get the projection of tensor onto onto
    proj = (
        t.sum(tensor * onto, dim=-1, keepdim=True)
        * onto
        / (t.norm(onto, dim=-1, keepdim=True) ** 2 + 1e-10)
    )
    # Subtract to get the orthogonal component
    return tensor - proj


def add_vector_after_position(
    matrix, vector, position_ids, after=None, do_projection=True
):
    after_id = after
    if after_id is None:
        after_id = position_ids.min().item() - 1

    mask = position_ids > after_id
    mask = mask.unsqueeze(-1)

    if do_projection:
        matrix = project_onto_orthogonal_complement(matrix, vector)

    matrix += mask.float() * vector
    return matrix


def find_subtensor_position(tensor, sub_tensor):
    n, m = tensor.size(0), sub_tensor.size(0)
    if m > n:
        return -1
    for i in range(n - m + 1):
        if t.equal(tensor[i : i + m], sub_tensor):
            return i
    return -1


def find_instruction_end_postion(tokens, end_str):
    end_pos = find_subtensor_position(tokens, end_str)
    return end_pos + len(end_str) - 1