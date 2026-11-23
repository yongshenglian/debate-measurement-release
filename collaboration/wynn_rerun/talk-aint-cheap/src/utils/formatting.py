import json
from typing import List, Optional
import colorama
from colorama import Fore, Style
from src.llm_api.base_llm_abstractions import Message, PRINT_COLORS

# Initialize colorama
colorama.init(autoreset=True)


def format_message_history(
    msg_history: List[Message],
    agent_name: Optional[str] = None,
    color_output: bool = True,
    indent: int = 2,
) -> str:
    """
    Format message history in a readable way.

    Args:
        msg_history: List of Message objects to format
        agent_name: Optional name of the agent to include in the header
        color_output: Whether to use colored output
        indent: Number of spaces to indent content

    Returns:
        Formatted string representation of the message history
    """
    if not msg_history:
        return "No message history available."

    formatted_output = []

    # Add header if agent name is provided
    if agent_name:
        header = f"Message History for {agent_name}"
        formatted_output.append(f"{header}\n{'=' * len(header)}")

    # Format each message
    for i, msg in enumerate(msg_history):
        # Get color for the role - fix the color mapping
        color_name = PRINT_COLORS.get(msg.role, "WHITE").upper()
        # Convert "LIGHT_GREEN" to "LIGHTGREEN_EX" and similar color names
        if color_name.startswith("LIGHT_"):
            color_name = color_name.replace("LIGHT_", "LIGHT") + "_EX"

        color_code = getattr(Fore, color_name) if color_output else ""
        reset_code = Style.RESET_ALL if color_output else ""

        # Format the role header
        role_header = f"{color_code}[{msg.role.upper()}]{reset_code}"

        # Format the content with indentation
        content_lines = msg.content.split("\n")
        indented_content = "\n".join([" " * indent + line for line in content_lines])

        # Add to output
        formatted_output.append(f"{role_header}\n{indented_content}")

        # Add separator between messages
        if i < len(msg_history) - 1:
            formatted_output.append("-" * 40)

    return "\n".join(formatted_output)


def print_message_history(
    msg_history: List[Message],
    agent_name: Optional[str] = None,
    color_output: bool = True,
    indent: int = 2,
) -> None:
    """
    Print message history in a readable format.

    Args:
        msg_history: List of Message objects to print
        agent_name: Optional name of the agent to include in the header
        color_output: Whether to use colored output
        indent: Number of spaces to indent content
    """
    formatted = format_message_history(
        msg_history=msg_history,
        agent_name=agent_name,
        color_output=color_output,
        indent=indent,
    )
    print(formatted)
