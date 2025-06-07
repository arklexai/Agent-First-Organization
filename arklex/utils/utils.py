"""Utility functions for the Arklex framework.

This module provides a collection of utility functions used throughout the Arklex framework,
including logging configuration, string manipulation, JSON processing, and chat history formatting.
These utilities support various aspects of the framework's functionality, from basic string
operations to complex data processing tasks.

The module is organized into several functional areas:
1. Logging: Configuration and setup of the logging system
2. Text Processing: String manipulation, tokenization, and similarity calculations
3. JSON Handling: Processing and validation of JSON data
4. Chat History: Formatting and management of conversation history

Key Features:
- Comprehensive logging setup with file rotation
- Text chunking and tokenization
- String similarity calculations
- JSON processing and validation
- Chat history formatting and truncation
"""

import os
import sys
import json
import logging
from logging.handlers import RotatingFileHandler
from typing import List, Dict, Any, Optional

import tiktoken
import Levenshtein

# Configure logging
logger: logging.Logger = logging.getLogger(__name__)


def init_logger(
    log_level: int = logging.INFO, filename: Optional[str] = None
) -> logging.Logger:
    """Initialize and configure the logging system.

    This function sets up logging with both file and stream handlers, configures log formatting,
    and suppresses noisy loggers. It creates the log directory if it doesn't exist.

    The function performs the following tasks:
    1. Configures the root logger
    2. Sets up file-based logging with rotation if a filename is provided
    3. Sets up console-based logging
    4. Configures log formatting and levels
    5. Suppresses noisy third-party loggers

    Logging Features:
    - File rotation with size limits (50MB per file)
    - Backup file management (20 backup files)
    - Formatted log messages with timestamps
    - Multiple output handlers (file and console)
    - Configurable log levels

    Args:
        log_level (int, optional): The logging level to use. Defaults to logging.INFO.
        filename (Optional[str], optional): Path to the log file. Defaults to None.

    Returns:
        logging.Logger: The configured logger instance.
    """
    root_logger: logging.Logger = logging.getLogger()  # Root logger

    # Remove existing handlers to reconfigure them
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    handlers: List[logging.Handler] = []
    # File handler
    if filename is not None:
        directory_name: str
        _: str
        directory_name, _ = os.path.split(filename)
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
        file_handler: RotatingFileHandler = RotatingFileHandler(
            filename=filename,
            mode="a",
            maxBytes=50 * 1024 * 1024,  # 50MB file size limit
            backupCount=20,  # Keep 20 backup files
            encoding=None,
            delay=0,
        )
        file_handler.setLevel(log_level)  # Set log level for the file
        file_handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
                datefmt="%m/%d/%Y %H:%M:%S",
            )
        )
        handlers.append(file_handler)

    # Stream (terminal) handler
    stream_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(log_level)  # Set log level for the terminal
    stream_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
        )
    )
    handlers.append(stream_handler)

    for handler in handlers:
        root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # Suppress noisy loggers
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)

    return logging.getLogger(__name__)


def chunk_string(
    text: str, tokenizer: str, max_length: int, from_end: bool = True
) -> str:
    """Chunk a string into tokens of specified maximum length.

    This function uses the specified tokenizer to split the input text into tokens
    and returns a chunk of the specified maximum length, either from the start or end.
    It's particularly useful for handling long text inputs in language models.

    The function:
    1. Initializes the specified tokenizer
    2. Encodes the input text into tokens
    3. Extracts the desired chunk based on the from_end parameter
    4. Decodes the tokens back into text

    Args:
        text (str): The input text to chunk.
        tokenizer (str): The name of the tokenizer to use (e.g., 'cl100k_base' for GPT models).
        max_length (int): The maximum number of tokens in the chunk.
        from_end (bool, optional): If True, take tokens from the end. Defaults to True.

    Returns:
        str: The chunked text string, containing at most max_length tokens.
    """
    # Initialize the tokenizer
    encoding: tiktoken.Encoding = tiktoken.get_encoding(tokenizer)
    tokens: List[int] = encoding.encode(text)
    chunks: str
    if from_end:
        chunks = encoding.decode(tokens[-max_length:])
    else:
        chunks = encoding.decode(tokens[:max_length])
    return chunks


def normalize(lst: List[float]) -> List[float]:
    """Normalize a list of numbers to sum to 1.

    This function takes a list of numbers and returns a new list where each number
    is divided by the sum of all numbers, effectively normalizing them to sum to 1.
    This is commonly used for converting raw scores into probabilities.

    The function:
    1. Calculates the sum of all numbers in the list
    2. Divides each number by the sum
    3. Returns the normalized list

    Args:
        lst (List[float]): The list of numbers to normalize.

    Returns:
        List[float]: The normalized list of numbers, where sum(result) = 1.
    """
    return [float(num) / sum(lst) for num in lst]


def str_similarity(string1: str, string2: str) -> float:
    """Calculate the similarity between two strings using Levenshtein distance.

    This function computes the similarity between two strings using the Levenshtein
    distance algorithm, normalized by the length of the longer string. The result
    is a score between 0 and 1, where:
    - 1.0 indicates identical strings
    - 0.0 indicates completely different strings

    The function:
    1. Calculates the Levenshtein distance between the strings
    2. Normalizes the distance by the length of the longer string
    3. Converts the distance to a similarity score
    4. Handles any errors gracefully

    Args:
        string1 (str): The first string to compare.
        string2 (str): The second string to compare.

    Returns:
        float: A similarity score between 0 and 1, where 1 indicates identical strings.
    """
    try:
        distance: int = Levenshtein.distance(string1, string2)
        max_length: int = max(len(string1), len(string2))
        similarity: float = 1 - (distance / max_length)
    except Exception as err:
        print(err)
        similarity = 0
    return similarity


def postprocess_json(raw_code: str) -> Optional[Dict[str, Any]]:
    """Process and validate raw JSON code.

    This function takes raw JSON code, filters out invalid lines, and attempts to
    parse it into a Python dictionary. It handles JSON decoding errors gracefully
    and provides detailed error logging when parsing fails.

    The function performs the following steps:
    1. Filters out lines that don't contain valid JSON syntax
    2. Joins the remaining lines
    3. Attempts to parse the resulting string as JSON
    4. Logs any errors that occur during parsing

    JSON Processing Features:
    - Validates JSON syntax
    - Filters invalid lines
    - Handles malformed JSON gracefully
    - Provides detailed error logging

    Args:
        raw_code (str): The raw JSON code to process.

    Returns:
        Optional[Dict[str, Any]]: The parsed JSON as a dictionary, or None if parsing fails.
    """
    valid_phrases: List[str] = ['"', "{", "}", "[", "]"]

    valid_lines: List[str] = []
    for line in raw_code.split("\n"):
        if len(line) == 0:
            continue
        # If the line not starts with any of the valid phrases, skip it
        should_skip: bool = not any(
            [line.strip().startswith(phrase) for phrase in valid_phrases]
        )
        if should_skip:
            continue
        valid_lines.append(line)

    try:
        generated_result: str = "\n".join(valid_lines)
        result: Optional[Dict[str, Any]] = json.loads(generated_result)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding generated JSON - {generated_result}")
        logger.error(f"raw result: {raw_code}")
        logger.error(f"Error: {e}")
        result = None
    return result


def truncate_string(text: str, max_length: int = 400) -> str:
    """Truncate a string to a maximum length.

    This function truncates the input string to the specified maximum length,
    adding an ellipsis if the string was truncated. This is useful for displaying
    long text in a limited space while indicating that the text was cut off.

    The function:
    1. Checks if the string exceeds the maximum length
    2. Truncates the string if necessary
    3. Adds an ellipsis to indicate truncation

    Args:
        text (str): The string to truncate.
        max_length (int, optional): The maximum length of the string. Defaults to 400.

    Returns:
        str: The truncated string, with "..." appended if truncation occurred.
    """
    if len(text) > max_length:
        text = text[:max_length] + "..."
    return text


def format_chat_history(chat_history: List[Dict[str, str]]) -> str:
    """Format chat history into a string representation.

    This function takes a list of chat messages and formats them into a single string,
    with each message prefixed by its role. The resulting format is:
    "role1: content1\nrole2: content2\n..."

    The function:
    1. Iterates through each message in the chat history
    2. Formats each message with its role and content
    3. Joins the messages with newlines
    4. Strips any trailing whitespace

    Args:
        chat_history (List[Dict[str, str]]): List of chat messages with 'role' and 'content'.

    Returns:
        str: The formatted chat history string, with each message on a new line.
    """
    chat_history_str: str = ""
    for turn in chat_history:
        chat_history_str += f"{turn['role']}: {turn['content']}\n"
    return chat_history_str.strip()


def format_truncated_chat_history(
    chat_history: List[Dict[str, str]], max_length: int = 400
) -> str:
    """Format chat history with truncated message content.

    This function takes a list of chat messages and formats them into a single string,
    with each message's content truncated to the specified maximum length. This is
    useful for displaying long conversations in a limited space.

    The function:
    1. Iterates through each message in the chat history
    2. Truncates each message's content if necessary
    3. Formats the messages with roles and truncated content
    4. Joins the messages with newlines
    5. Strips any trailing whitespace

    Args:
        chat_history (List[Dict[str, str]]): List of chat messages with 'role' and 'content'.
        max_length (int, optional): Maximum length for each message. Defaults to 400.

    Returns:
        str: The formatted chat history string with truncated messages.
    """
    chat_history_str: str = ""
    for turn in chat_history:
        chat_history_str += f"{turn['role']}: {truncate_string(turn['content'], max_length) if turn['content'] else turn['content']}\n"
    return chat_history_str.strip()
