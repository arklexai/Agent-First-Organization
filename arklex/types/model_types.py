from pydantic import BaseModel


class LLMConfig(BaseModel):
    """Configuration for language model settings.

    This class defines the configuration parameters for language models used in the system.
    It specifies which model to use and from which provider.

    The class provides:
    1. Model selection and configuration
    2. Provider specification
    3. Type-safe configuration management

    Attributes:
        model_type_or_path (str): The model identifier or path to use.
        llm_provider (str): The provider of the language model (e.g., 'openai', 'anthropic').
    """

    model_type_or_path: str
    llm_provider: str
