import torch as t
from typing import Optional
import os
from dataclasses import dataclass


@dataclass
class SteeringSettings:
    """
    max_new_tokens: Maximum number of tokens to generate.
    type: Type of test to run. One of "in_distribution", "out_of_distribution", "truthful_qa".
    few_shot: Whether to test with few-shot examples in the prompt. One of "positive", "negative", "none".
    do_projection: Whether to project activations onto orthogonal complement of steering vector.
    override_vector: If not None, the steering vector generated from this layer's activations will be used at all layers. Use to test the effect of steering with a different layer's vector.
    override_vector_model: If not None, steering vectors generated from this model will be used instead of the model being used for generation - use to test vector transference between models.
    use_base_model: Whether to use the base model instead of the chat model.
    model_size: Size of the model to use. One of "7b", "13b".
    n_test_datapoints: Number of datapoints to test on. If None, all datapoints will be used.
    add_every_token_position: Whether to add steering vector to every token position (including question), not only to token positions corresponding to the model's response to the user
    override_model_weights_path: If not None, the model weights at this path will be used instead of the model being used for generation - use to test using activation steering on top of fine-tuned model.
    """

    max_new_tokens: int = 100
    type: str = "in_distribution"
    few_shot: str = "none"
    do_projection: bool = False
    override_vector: Optional[int] = None
    override_vector_model: Optional[str] = None
    use_base_model: bool = False
    model_size: str = "7b"
    n_test_datapoints: Optional[int] = None
    add_every_token_position: bool = False
    override_model_weights_path: Optional[str] = None

    def make_result_save_suffix(
        self,
        layer: Optional[int] = None,
        multiplier: Optional[int] = None,
    ):
        elements = {
            "layer": layer,
            "multiplier": multiplier,
            "max_new_tokens": self.max_new_tokens,
            "type": self.type,
            "few_shot": self.few_shot,
            "do_projection": self.do_projection,
            "override_vector": self.override_vector,
            "override_vector_model": self.override_vector_model,
            "use_base_model": self.use_base_model,
            "model_size": self.model_size,
            "n_test_datapoints": self.n_test_datapoints,
            "add_every_token_position": self.add_every_token_position,
            "override_model_weights_path": self.override_model_weights_path,
        }
        return "_".join([f"{k}={str(v).replace('/', '-')}" for k, v in elements.items() if v is not None])

    def filter_result_files_by_suffix(
        self,
        directory: str,
        layer: Optional[int] = None,
        multiplier: Optional[int] = None,
    ):
        elements = {
            "layer": layer,
            "multiplier": multiplier,
            "max_new_tokens": self.max_new_tokens,
            "type": self.type,
            "few_shot": self.few_shot,
            "do_projection": self.do_projection,
            "override_vector": self.override_vector,
            "override_vector_model": self.override_vector_model,
            "use_base_model": self.use_base_model,
            "model_size": self.model_size,
            "n_test_datapoints": self.n_test_datapoints,
            "add_every_token_position": self.add_every_token_position,
            "override_model_weights_path": self.override_model_weights_path,
        }

        filtered_elements = {k: v for k, v in elements.items() if v is not None}
        remove_elements = {k for k, v in elements.items() if v is None}

        matching_files = []

        print(self.override_model_weights_path)

        for filename in os.listdir(directory):
            if all(f"{k}={str(v).replace('/', '-')}" in filename for k, v in filtered_elements.items()):
                # ensure remove_elements are *not* present
                if all(f"{k}=" not in filename for k in remove_elements):
                    matching_files.append(filename)

        return [os.path.join(directory, f) for f in matching_files]


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


def find_last_subtensor_position(tensor, sub_tensor):
    n, m = tensor.size(0), sub_tensor.size(0)
    if m > n:
        return -1
    for i in range(n - m, -1, -1):
        if t.equal(tensor[i : i + m], sub_tensor):
            return i
    return -1


def find_instruction_end_postion(tokens, end_str):
    start_pos = find_last_subtensor_position(tokens, end_str)
    if start_pos == -1:
        return -1
    return start_pos + len(end_str) - 1


def get_a_b_probs(logits, a_token_id, b_token_id):
    last_token_logits = logits[0, -1, :]
    last_token_probs = t.softmax(last_token_logits, dim=-1)
    a_prob = last_token_probs[a_token_id].item()
    b_prob = last_token_probs[b_token_id].item()
    return a_prob, b_prob


def make_tensor_save_suffix(layer, model_name_path):
    return f'{layer}_{model_name_path.split("/")[-1]}'
