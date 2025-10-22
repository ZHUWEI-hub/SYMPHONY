"""Factory functions for creating generators and models."""

from .py_generate import PyGenerator
from .rs_generate import RsGenerator
from .go_generate import GoGenerator
from .generator_types import Generator
from .model import CodeLlama, ModelBase, GPT4, GPT35, StarChat, GPTDavinci


def generator_factory(lang: str) -> Generator:
    """Create a code generator for the specified language.
    
    Args:
        lang: Programming language ('py', 'python', 'rs', 'rust', 'go', 'golang')
        
    Returns:
        Generator instance for the specified language
        
    Raises:
        ValueError: If language is not supported
    """
    if lang == "py" or lang == "python":
        return PyGenerator()
    elif lang == "rs" or lang == "rust":
        return RsGenerator()
    elif lang == "go" or lang == "golang":
        return GoGenerator()
    else:
        raise ValueError(f"Invalid language for generator: {lang}")


def model_factory(model_name: str) -> ModelBase:
    """Create a model instance for the specified model name.
    
    Args:
        model_name: Name of the model (e.g., 'gpt-4', 'gpt-3.5-turbo', 'starchat', 'codellama', etc.)
        
    Returns:
        ModelBase instance for the specified model
        
    Raises:
        ValueError: If model name is not supported
    """
    if model_name == "gpt-4" or model_name == "Mistral-7B-Instruct-v0.3":
        return GPT4()
    elif model_name == "gpt-3.5-turbo":
        return GPT35()
    elif model_name == "starchat":
        return StarChat()
    elif model_name.startswith("codellama"):
        kwargs = {}
        if "-" in model_name:
            kwargs["version"] = model_name.split("-")[1]
        return CodeLlama(**kwargs)
    elif model_name.startswith("text-davinci"):
        return GPTDavinci(model_name)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
