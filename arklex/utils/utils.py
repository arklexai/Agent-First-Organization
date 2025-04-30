import os
import sys
import json
import logging
from logging.handlers import RotatingFileHandler

import tiktoken
import Levenshtein

logger = logging.getLogger(__name__)


def init_logger(log_level=logging.INFO, filename=None):
    root_logger = logging.getLogger()  # Root logger

    # Remove existing handlers to reconfigure them
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    handlers = []
    # File handler
    if filename is not None:
        directory_name, _ = os.path.split(filename)
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
        file_handler = RotatingFileHandler(
            filename=filename, 
            mode='a',
            maxBytes=50*1024*1024,
            backupCount=20,
            encoding=None,
            delay=0
        )
        file_handler.setLevel(log_level)  # Set log level for the file
        file_handler.setFormatter(logging.Formatter(
            "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S"
        ))
        handlers.append(file_handler)

    # Stream (terminal) handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(log_level)  # Set log level for the terminal
    stream_handler.setFormatter(logging.Formatter(
        "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S"
    ))
    handlers.append(stream_handler)

    for handler in handlers:
        root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # Suppress noisy loggers
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)

    return logging.getLogger(__name__)


def chunk_string(text, tokenizer, max_length, from_end=True):
    # Initialize the tokenizer
	encoding = tiktoken.get_encoding(tokenizer)
	tokens = encoding.encode(text)
	if from_end:
		chunks = encoding.decode(tokens[-max_length:])
	else:
		chunks = encoding.decode(tokens[:max_length])
	return chunks

def normalize(lst):
		return [float(num)/sum(lst) for num in lst]

def str_similarity(string1, string2):
	try:
		distance = Levenshtein.distance(string1, string2)
		max_length = max(len(string1), len(string2))
		similarity = 1 - (distance / max_length)
	except Exception as err:
		print(err)
		similarity = 0
	return similarity


def postprocess_json(raw_code):
	valid_phrases = ['"', '{', '}', '[', ']']

	valid_lines = []
	for line in raw_code.split('\n'):
		if len(line) == 0:
			continue
		# If the line not starts with any of the valid phrases, skip it
		should_skip = not any([line.strip().startswith(phrase) for phrase in valid_phrases])
		if should_skip:
			continue
		valid_lines.append(line)

	try:
		generated_result = "\n".join(valid_lines)
		result = json.loads(generated_result)
	except json.JSONDecodeError as e:
		logger.error(f"Error decoding generated JSON - {generated_result}")
		logger.error(f"raw result: {raw_code}")
		logger.error(f"Error: {e}")
		result = None
	return result

def truncate_string(text: str, max_length: int=400):
    if len(text) > max_length:
        text = text[:max_length] + "..."
    return text

def format_chat_history(chat_history):
    '''Includes current user utterance'''
    chat_history_str= ""
    for turn in chat_history:
        chat_history_str += f"{turn['role']}: {turn['content']}\n"
    return chat_history_str.strip()

def format_truncated_chat_history(chat_history, max_length=400):
    '''Includes current user utterance'''
    chat_history_str= ""
    for turn in chat_history:
        chat_history_str += f"{turn['role']}: {truncate_string(turn['content'], max_length) if turn['content'] else turn['content']}\n"
    return chat_history_str.strip()


   