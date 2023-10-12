from typing import List, Tuple, Optional


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def tokenize_llama(
    tokenizer,
    system_prompt: str,
    conversation: List[Tuple[str, Optional[str]]],
    no_final_eos=False,
    chat_model=True,
) -> List[int]:
    """
    tokenizer: a HuggingFace tokenizer
    system_prompt: the system prompt to use for the conversation
    conversation: a list of tuples of the form (user_input, model_output) - model_output can be None if there is no response yet
    no_final_eos: if True, the final </s> token will not be added to the end of the conversation when there is a model response
    chat_model: whether input is meant for a llama chat model or a llama base model (True means chat model, False means base model)

    Returns: a list of tokens
    """
    if chat_model:
        return tokenize_llama_chat(tokenizer, system_prompt, conversation, no_final_eos)
    else:
        return tokenize_llama_base(tokenizer, conversation, no_final_eos)


def tokenize_llama_chat(
    tokenizer,
    system_prompt: str,
    conversation: List[Tuple[str, Optional[str]]],
    no_final_eos=False,
) -> List[int]:
    """
    tokenizer: a HuggingFace tokenizer
    system_prompt: the system prompt to use for the conversation
    conversation: a list of tuples of the form (user_input, model_output) - model_output can be None if there is no response yet
    no_final_eos: if True, the final </s> token will not be added to the end of the conversation when there is a model response

    Returns: a list of tokens
    """

    def _instruction_response_to_tokens(
        instruction, model_output=None, is_first_message=False, no_eos=False
    ):
        if is_first_message:
            dialog_content = B_SYS + system_prompt + E_SYS + instruction.strip()
        else:
            dialog_content = instruction.strip()
        if model_output is not None:
            if no_eos:
                return tokenizer.encode(
                    f"{B_INST} {dialog_content.strip()} {E_INST} {model_output.strip()}"
                )
            return tokenizer.encode(
                f"{B_INST} {dialog_content.strip()} {E_INST} {model_output.strip()} {tokenizer.eos_token}"
            )
        else:
            return tokenizer.encode(f"{B_INST} {dialog_content.strip()} {E_INST}")

    tokens = []
    for i, (user_input, model_output) in enumerate(conversation):
        tokens += _instruction_response_to_tokens(
            user_input,
            model_output,
            i == 0,
            no_final_eos and (i == len(conversation) - 1),
        )
    return tokens


def tokenize_llama_base(
    tokenizer, conversation: List[Tuple[str, Optional[str]]], no_final_eos=False
) -> List[int]:
    """
    tokenizer: a HuggingFace tokenizer
    conversation: a list of tuples of the form (user_input, model_output) - model_output can be None if there is no response yet
    no_final_eos: if True, the final </s> token will not be added to the end of the conversation when there is a model response

    Returns: a list of tokens
    """
    if len(conversation) == 0:
        return []
    full_text = []
    for user_input, model_output in conversation:
        text = f"Input: {user_input.strip()}"
        if model_output:
            text += f"\nResponse: {model_output.strip()}"
        full_text.append(text)
    full_text = "\n\n".join(full_text)
    if not no_final_eos and conversation[-1][1] is not None:
        full_text += f" {tokenizer.eos_token}"
    return tokenizer.encode(full_text)
