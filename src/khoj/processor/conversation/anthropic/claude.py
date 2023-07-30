# Standard Packages
import logging
from datetime import datetime
from typing import Optional

# External Packages
from langchain.schema import ChatMessage

# Internal Packages
from khoj.utils.constants import empty_escape_sequences
from khoj.processor.conversation import prompts
from khoj.processor.conversation.anthropic.utils import (
    chat_completion_with_backoff,
    completion_with_backoff,
)
from khoj.processor.conversation.utils import generate_chatml_messages_with_context


logger = logging.getLogger(__name__)


def extract_questions_anthropic(
    text, model: Optional[str] = "claude-instant-1", conversation_log={}, api_key=None, temperature=0, max_tokens=100
):
    """
    Infer search queries to retrieve relevant notes to answer user query
    """
    # Extract Past User Message and Inferred Questions from Conversation Log
    chat_history = ""

    for chat in conversation_log.get("chat", [])[-4:]:
        if chat["by"] == "khoj":
            chat_history += f"Human: {chat['intent']['query']}\n"
            chat_history += f"Assisstant: {chat['message']}\n"

    # Get dates relative to today for prompt creation
    today = datetime.today()
    current_new_year = today.replace(month=1, day=1)
    last_new_year = current_new_year.replace(year=today.year - 1)

    prompt = prompts.extract_questions_claude.format(
        current_date=today.strftime("%A, %Y-%m-%d"),
        last_new_year=last_new_year.strftime("%Y"),
        last_new_year_date=last_new_year.strftime("%Y-%m-%d"),
        current_new_year_date=current_new_year.strftime("%Y-%m-%d"),
        bob_tom_age_difference={current_new_year.year - 1984 - 30},
        bob_age={current_new_year.year - 1984},
        chat_history=chat_history,
        text=text,
    )
    messages = [ChatMessage(content=prompt, role="human")]

    # Get Response from GPT
    response = completion_with_backoff(
        messages=messages,
        model_name=model,
        temperature=temperature,
        max_tokens=max_tokens,
        model_kwargs={"stop": ["A: ", "\n"]},
        anthropic_api_key=api_key,
    )

    # Extract, Clean Message from Claude's Response
    try:
        questions = (
            response.content.strip(empty_escape_sequences)
            .replace("['", '["')
            .replace("']", '"]')
            .replace("', '", '", "')
            .replace('["', "")
            .replace('"]', "")
            .split('", "')
        )
    except:
        logger.warning(f"Claude returned invalid JSON. Falling back to using user message as search query.\n{response}")
        questions = [text]
    logger.debug(f"Extracted Questions by Claude: {questions}")
    return questions


def converse_anthropic(
    references,
    user_query,
    conversation_log={},
    model: str = "claude-instant-1",
    api_key: Optional[str] = None,
    temperature: float = 0.2,
    completion_func=None,
):
    """
    Converse with user using Anthropic's Claude
    """
    # Initialize Variables
    current_date = datetime.now().strftime("%Y-%m-%d")
    compiled_references = "\n\n".join({f"# {item}" for item in references})

    # Get Conversation Primer appropriate to Conversation Type
    if compiled_references == "":
        conversation_primer = prompts.general_conversation_claude.format(current_date=current_date, query=user_query)
    else:
        conversation_primer = prompts.notes_conversation_claude.format(
            current_date=current_date, query=user_query, references=compiled_references
        )

    # Setup Prompt with Primer or Conversation History
    messages = generate_chatml_messages_with_context(
        conversation_primer,
        prompts.personality_claude.format(),
        conversation_log,
        model,
    )
    truncated_messages = "\n".join({f"{message.content[:40]}..." for message in messages})
    logger.debug(f"Conversation Context for Claude: {truncated_messages}")

    # Get Response from Claude
    return chat_completion_with_backoff(
        messages=messages,
        compiled_references=references,
        model_name=model,
        temperature=temperature,
        anthropic_api_key=api_key,
        completion_func=completion_func,
    )
