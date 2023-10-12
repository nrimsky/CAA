import pytest
from tokenize_llama import tokenize_llama, B_SYS, E_SYS, B_INST, E_INST
from unittest.mock import Mock

system_prompt = "You are a helpful, respectful and honest assistant."


@pytest.fixture
def mock_tokenizer():
    mock = Mock()
    mock.encode = lambda x: [c for c in "<s> " + x]
    mock.eos_token = "</s>"
    return mock


def test_tokenize_llama_empty_conversation(mock_tokenizer):
    conversation = []
    tokens = tokenize_llama(mock_tokenizer, system_prompt, conversation)
    assert "".join(tokens) == ""


def test_tokenize_llama_single_message_no_response(mock_tokenizer):
    conversation = [("Hi, how are you?", None)]
    expected_output = (
        f"<s> {B_INST} {B_SYS}{system_prompt}{E_SYS}Hi, how are you? {E_INST}"
    )
    tokens = tokenize_llama(mock_tokenizer, system_prompt, conversation)
    assert "".join(tokens) == expected_output


def test_tokenize_llama_single_message_with_response(mock_tokenizer):
    conversation = [("Hi, how are you?", "I'm fine, thank you.")]
    expected_output = f"<s> {B_INST} {B_SYS}{system_prompt}{E_SYS}Hi, how are you? {E_INST} I'm fine, thank you. </s>"
    tokens = tokenize_llama(mock_tokenizer, system_prompt, conversation)
    assert "".join(tokens) == expected_output


def test_tokenize_llama_multiple_messages(mock_tokenizer):
    conversation = [
        ("Hi, how are you?", "I'm fine, thank you."),
        ("What's your name?", "I'm Assistant."),
    ]
    tokens = tokenize_llama(mock_tokenizer, system_prompt, conversation)
    expected_output = f"<s> {B_INST} {B_SYS}{system_prompt}{E_SYS}Hi, how are you? {E_INST} I'm fine, thank you. </s><s> {B_INST} What's your name? {E_INST} I'm Assistant. </s>"
    assert "".join(tokens) == expected_output


def test_tokenize_llama_multiple_messages_with_empty_response(mock_tokenizer):
    conversation = [
        ("Hi, how are you?", "I'm fine, thank you."),
        ("What's your name?", None),
    ]
    tokens = tokenize_llama(mock_tokenizer, system_prompt, conversation)
    expected_output = f"<s> {B_INST} {B_SYS}{system_prompt}{E_SYS}Hi, how are you? {E_INST} I'm fine, thank you. </s><s> {B_INST} What's your name? {E_INST}"
    assert "".join(tokens) == expected_output


def test_tokenize_llama_multiple_messages_no_eos(mock_tokenizer):
    conversation = [
        ("Hi, how are you?", "I'm fine, thank you."),
        ("What's your name?", "I'm Assistant."),
    ]
    tokens = tokenize_llama(
        mock_tokenizer, system_prompt, conversation, no_final_eos=True
    )
    expected_output = f"<s> {B_INST} {B_SYS}{system_prompt}{E_SYS}Hi, how are you? {E_INST} I'm fine, thank you. </s><s> {B_INST} What's your name? {E_INST} I'm Assistant."
    assert "".join(tokens) == expected_output


def test_tokenize_llama_single_input_no_eos(mock_tokenizer):
    conversation = [("Hi, how are you?", "I'm fine, thank you.")]
    tokens = tokenize_llama(
        mock_tokenizer, system_prompt, conversation, no_final_eos=True
    )
    expected_output = f"<s> {B_INST} {B_SYS}{system_prompt}{E_SYS}Hi, how are you? {E_INST} I'm fine, thank you."
    assert "".join(tokens) == expected_output


def test_tokenize_llama_base_empty_conversation(mock_tokenizer):
    conversation = []
    tokens = tokenize_llama(mock_tokenizer, "", conversation, chat_model=False)
    assert "".join(tokens) == ""


def test_tokenize_llama_base_single_message_no_response(mock_tokenizer):
    conversation = [("Hi, how are you?", None)]
    expected_output = f"<s> Input: Hi, how are you?"
    tokens = tokenize_llama(mock_tokenizer, "", conversation, chat_model=False)
    assert "".join(tokens) == expected_output


def test_tokenize_llama_base_single_message_with_response(mock_tokenizer):
    conversation = [("Hi, how are you?", "I'm fine, thank you.")]
    expected_output = (
        f"<s> Input: Hi, how are you?\nResponse: I'm fine, thank you. </s>"
    )
    tokens = tokenize_llama(mock_tokenizer, "", conversation, chat_model=False)
    assert "".join(tokens) == expected_output


def test_tokenize_llama_base_multiple_messages(mock_tokenizer):
    conversation = [
        ("Hi, how are you?", "I'm fine, thank you."),
        ("What's your name?", "I'm Assistant."),
    ]
    expected_output = f"<s> Input: Hi, how are you?\nResponse: I'm fine, thank you.\n\nInput: What's your name?\nResponse: I'm Assistant. </s>"
    tokens = tokenize_llama(mock_tokenizer, "", conversation, chat_model=False)
    assert "".join(tokens) == expected_output


def test_tokenize_llama_base_multiple_messages_with_empty_response(mock_tokenizer):
    conversation = [
        ("Hi, how are you?", "I'm fine, thank you."),
        ("What's your name?", None),
    ]
    expected_output = f"<s> Input: Hi, how are you?\nResponse: I'm fine, thank you.\n\nInput: What's your name?"
    tokens = tokenize_llama(mock_tokenizer, "", conversation, chat_model=False)
    assert "".join(tokens) == expected_output


def test_tokenize_llama_base_multiple_messages_no_eos(mock_tokenizer):
    conversation = [
        ("Hi, how are you?", "I'm fine, thank you."),
        ("What's your name?", "I'm Assistant."),
    ]
    tokens = tokenize_llama(
        mock_tokenizer, "", conversation, no_final_eos=True, chat_model=False
    )
    expected_output = f"<s> Input: Hi, how are you?\nResponse: I'm fine, thank you.\n\nInput: What's your name?\nResponse: I'm Assistant."
    assert "".join(tokens) == expected_output


def test_tokenize_llama_base_single_input_no_eos(mock_tokenizer):
    conversation = [("Hi, how are you?", "I'm fine, thank you.")]
    tokens = tokenize_llama(
        mock_tokenizer, "", conversation, no_final_eos=True, chat_model=False
    )
    expected_output = f"<s> Input: Hi, how are you?\nResponse: I'm fine, thank you."
    assert "".join(tokens) == expected_output
