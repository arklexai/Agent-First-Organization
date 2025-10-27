import Levenshtein


def str_similarity(string1: str, string2: str) -> float:
    try:
        distance: int = Levenshtein.distance(string1, string2)
        max_length: int = max(len(string1), len(string2))
        similarity: float = 1 - (distance / max_length)
    except Exception as err:
        print(err)
        similarity = 0
    return similarity


def format_chat_history(chat_history: list[dict[str, str]]) -> str:
    lines = []

    for turn in chat_history:
        role = turn.get("role")
        content = turn.get("content", "")
        type_ = turn.get("type")

        if type_ == "function_call":
            name = turn.get("name", "unknown_function")
            args = turn.get("arguments", "{}")
            lines.append(f"function_call ({name}): {args}")

        elif type_ == "function_call_output":
            output = turn.get("output", "")
            lines.append(f"function_call_output: {output}")

        elif role in {"assistant", "user"}:
            lines.append(f"{role}: {content}")

        else:
            # Fallback for unknown entries
            lines.append(f"{role or type_}: {content}")

    return "\n".join(lines).strip()
