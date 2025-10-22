"""Model interfaces and OpenAI API wrappers."""

import json
import logging
import os
import dataclasses
from typing import List, Union, Optional, Literal

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import openai


MessageRole = Literal["system", "user", "assistant"]

openai.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
openai.api_key = os.getenv("OPENAI_API_KEY", "")

task_status = {}

@dataclasses.dataclass()
class Message:
    """Message in a chat conversation."""
    role: MessageRole
    content: str


def message_to_str(message: Message) -> str:
    """Convert a message to string format."""
    return f"{message.role}: {message.content}"


def messages_to_str(messages: List[Message]) -> str:
    """Convert a list of messages to string format."""
    return "\n".join([message_to_str(message) for message in messages])


SEED_FILE = "task_id.json"


def load_seed():
    """Load the current task seed from file."""
    if os.path.exists(SEED_FILE):
        with open(SEED_FILE, 'r') as f:
            return json.load(f)["current_seed"]
    return 0  

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def gpt_completion(
    model: str,
    prompt: str,
    max_tokens: int = 1024,
    stop_strs: Optional[List[str]] = None,
    temperature: float = 0.0,
    num_comps=1,
) -> Union[List[str], str]:
    """Generate completion using OpenAI API with multi-agent selection.
    
    Args:
        model: Model name (used for fallback)
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        stop_strs: Stop sequences
        temperature: Sampling temperature
        num_comps: Number of completions
        
    Returns:
        Generated completion(s)
    """
    from programming.run import llm_manager

    selected_model = llm_manager.call_llm()
    print(f"Using model: {selected_model}")
    
    response = openai.Completion.create(
        model=selected_model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop_strs,
        n=num_comps,
    )
    
    if num_comps == 1:
        return response.choices[0].text  # type: ignore
    return [choice.text for choice in response.choices]  # type: ignore


@retry(wait=wait_random_exponential(min=1, max=180), stop=stop_after_attempt(6))
def gpt_chat_reflection(
    model: str,
    messages: List[Message],
    max_tokens: int = 1024,
    temperature: float = 0.0,
    num_comps=1,
) -> Union[List[str], str]:
    """Generate chat completion for reflection using multi-agent selection.
    
    Args:
        model: Model name (used for fallback)
        messages: List of chat messages
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        num_comps: Number of completions
        
    Returns:
        Generated completion(s)
    """
    from programming.run import llm_manager

    task_id = load_seed()
    selected_model = llm_manager.call_llm()
    print(f"Reflection model: {selected_model}")
    
    response = openai.ChatCompletion.create(
        model=selected_model,
        messages=[dataclasses.asdict(message) for message in messages],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        n=num_comps,
    )
    
    if num_comps == 1:
        return response.choices[0].message.content  # type: ignore
    return [choice.message.content for choice in response.choices]  # type: ignore


@retry(wait=wait_random_exponential(min=1, max=180), stop=stop_after_attempt(6))
def gpt_chat(
    model: str,
    messages: List[Message],
    max_tokens: int = 1024,
    temperature: float = 0.0,
    num_comps=1,
) -> Union[List[str], str]:
    """Generate chat completion using multi-agent selection.
    
    Args:
        model: Model name (used for fallback)
        messages: List of chat messages
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        num_comps: Number of completions
        
    Returns:
        List containing [response, model_name] for single completion,
        or list of responses for multiple completions
    """
    from programming.run import llm_manager

    task_id = load_seed()
    selected_model = llm_manager.call_llm()
    print(f"Using model: {selected_model}")
    
    response = openai.ChatCompletion.create(
        model=selected_model,
        messages=[dataclasses.asdict(message) for message in messages],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        n=num_comps,
    )
    
    if num_comps == 1:
        return [response.choices[0].message.content, selected_model]  # type: ignore
    return [choice.message.content for choice in response.choices]  # type: ignore


class ModelBase:
    """Base class for all models."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_chat = False

    def __repr__(self) -> str:
        return f'{self.name}'

    def generate_chat(
        self, 
        messages: List[Message], 
        max_tokens: int = 1024, 
        temperature: float = 0.2, 
        num_comps: int = 1
    ) -> Union[List[str], str]:
        """Generate chat completion."""
        raise NotImplementedError

    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 1024, 
        stop_strs: Optional[List[str]] = None, 
        temperature: float = 0.0, 
        num_comps=1
    ) -> Union[List[str], str]:
        """Generate completion."""
        raise NotImplementedError


class GPTChat(ModelBase):
    """Base class for GPT chat models."""
    
    def __init__(self, model_name: str):
        self.name = model_name
        self.is_chat = True

    def generate_chat(
        self, 
        messages: List[Message], 
        max_tokens: int = 1024, 
        temperature: float = 0.2, 
        num_comps: int = 1
    ) -> Union[List[str], str]:
        """Generate chat completion."""
        return gpt_chat(self.name, messages, max_tokens, temperature, num_comps)

    def generate_reflection(
        self, 
        messages: List[Message], 
        max_tokens: int = 1024, 
        temperature: float = 0.2, 
        num_comps: int = 1
    ) -> Union[List[str], str]:
        """Generate reflection completion."""
        return gpt_chat_reflection(self.name, messages, max_tokens, temperature, num_comps)


class GPT4(GPTChat):
    """GPT-4 model wrapper."""
    
    def __init__(self):
        super().__init__("Mistral-7B-Instruct-v0.3")


class GPT35(GPTChat):
    """GPT-3.5-Turbo model wrapper."""
    
    def __init__(self):
        super().__init__("gpt-3.5-turbo")


class GPTDavinci(ModelBase):
    """GPT-3 Davinci model wrapper."""
    
    def __init__(self, model_name: str):
        self.name = model_name

    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 1024, 
        stop_strs: Optional[List[str]] = None, 
        temperature: float = 0, 
        num_comps=1
    ) -> Union[List[str], str]:
        """Generate completion."""
        return gpt_completion(self.name, prompt, max_tokens, stop_strs, temperature, num_comps)


class HFModelBase(ModelBase):
    """Base class for HuggingFace chat models."""

    def __init__(self, model_name: str, model, tokenizer, eos_token_id=None):
        self.name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id
        self.is_chat = True

    def generate_chat(
        self, 
        messages: List[Message], 
        max_tokens: int = 1024, 
        temperature: float = 0.2, 
        num_comps: int = 1
    ) -> Union[List[str], str]:
        """Generate chat completion using HuggingFace model."""
        if temperature < 0.0001:
            temperature = 0.0001

        prompt = self.prepare_prompt(messages)

        outputs = self.model.generate(
            prompt,
            max_new_tokens=min(max_tokens, self.model.config.max_position_embeddings),
            use_cache=True,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            eos_token_id=self.eos_token_id,
            num_return_sequences=num_comps,
        )

        outs = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        assert isinstance(outs, list)
        for i, out in enumerate(outs):
            assert isinstance(out, str)
            outs[i] = self.extract_output(out)

        if len(outs) == 1:
            return outs[0]  # type: ignore
        return outs  # type: ignore

    def prepare_prompt(self, messages: List[Message]):
        """Prepare prompt from messages (must be implemented by subclass)."""
        raise NotImplementedError

    def extract_output(self, output: str) -> str:
        """Extract output from model response (must be implemented by subclass)."""
        raise NotImplementedError


class StarChat(HFModelBase):
    """StarChat model wrapper."""
    
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceH4/starchat-beta",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/starchat-beta")
        super().__init__("starchat", model, tokenizer, eos_token_id=49155)

    def prepare_prompt(self, messages: List[Message]):
        """Prepare prompt in StarChat format."""
        prompt = ""
        for i, message in enumerate(messages):
            prompt += f"<|{message.role}|>\n{message.content}\n<|end|>\n"
            if i == len(messages) - 1:
                prompt += "<|assistant|>\n"
        return self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

    def extract_output(self, output: str) -> str:
        """Extract output from StarChat response."""
        out = output.split("<|assistant|>")[1]
        if out.endswith("<|end|>"):
            out = out[:-len("<|end|>")]
        return out


class CodeLlama(HFModelBase):
    """CodeLlama model wrapper."""
    
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. \
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. \
Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something \
not correct. If you don't know the answer to a question, please don't share false information."""

    def __init__(self, version: Literal["34b", "13b", "7b"] = "34b"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(
            f"codellama/CodeLlama-{version}-Instruct-hf",
            add_eos_token=True,
            add_bos_token=True,
            padding_side='left'
        )
        model = AutoModelForCausalLM.from_pretrained(
            f"codellama/CodeLlama-{version}-Instruct-hf",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        super().__init__("codellama", model, tokenizer)

    def prepare_prompt(self, messages: List[Message]):
        """Prepare prompt in CodeLlama format."""
        if messages[0].role != "system":
            messages = [
                Message(role="system", content=self.DEFAULT_SYSTEM_PROMPT)
            ] + messages
        messages = [
            Message(
                role=messages[1].role, 
                content=self.B_SYS + messages[0].content + self.E_SYS + messages[1].content
            )
        ] + messages[2:]
        
        assert all([msg.role == "user" for msg in messages[::2]]) and all(
            [msg.role == "assistant" for msg in messages[1::2]]
        ), (
            "model only supports 'system', 'user' and 'assistant' roles, "
            "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
        )
        
        messages_tokens: List[int] = sum(
            [
                self.tokenizer.encode(
                    f"{self.B_INST} {(prompt.content).strip()} {self.E_INST} {(answer.content).strip()} ",
                )
                for prompt, answer in zip(messages[::2], messages[1::2])
            ],
            [],
        )
        
        assert messages[-1].role == "user", f"Last message must be from user, got {messages[-1].role}"
        messages_tokens += self.tokenizer.encode(
            f"{self.B_INST} {(messages[-1].content).strip()} {self.E_INST}",
        )
        messages_tokens = messages_tokens[:-1]
        
        import torch
        return torch.tensor([messages_tokens]).to(self.model.device)

    def extract_output(self, output: str) -> str:
        """Extract output from CodeLlama response."""
        out = output.split("[/INST]")[-1].split("</s>")[0].strip()
        return out
