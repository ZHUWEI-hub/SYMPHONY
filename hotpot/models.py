"""OpenAI API interface for language model calls."""

import logging
import os
import random
import time
import warnings

import openai
import backoff
from transformers import GPT2Tokenizer


completion_tokens = prompt_tokens = 0
MAX_TOKENS = 4000

# Configure OpenAI API
openai.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
openai.api_key = os.getenv("OPENAI_API_KEY", "")

def tokens_in_text(text):
    """Placeholder function for token counting.
    
    Args:
        text: The input text
        
    Returns:
        Fixed token count (placeholder implementation)
    """
    return 712

def gpt3(prompt, model="gpt-3.5-turbo-16k", temperature=0, max_tokens=100, n=1, stop=None) -> list:
    """Call GPT-3.5 for evaluation and reflection tasks.
    
    Args:
        prompt: The prompt to send to the model
        model: Model name to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        n: Number of completions to generate
        stop: Stop sequences
        
    Returns:
        List of generated text completions
    """
    from run import llm_manager
    print("Generating evaluation reflection")
    messages = [{"role": "user", "content": prompt}]
    global completion_tokens, prompt_tokens
    outputs = []
    
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        for i in range(cnt):
            try:
                selected_model = llm_manager.call_llm()
                print(f"Using model: {selected_model}")
                
                res = completions_with_backoff(
                    model=selected_model, 
                    messages=messages, 
                    temperature=temperature,
                    max_tokens=200, 
                    n=1, 
                    stop=stop
                )
                outputs.extend([choice["message"]["content"] for choice in res["choices"]])

                text_temp = [choice["message"]["content"] for choice in res["choices"]]

                completion_tokens += res["usage"]["completion_tokens"]
                prompt_tokens += res["usage"]["prompt_tokens"]
                logging.info(f"Model: {selected_model}, Response: {text_temp}")
                logging.info(f"Completion tokens: {completion_tokens}, Prompt tokens: {prompt_tokens}")
            except openai.error.OpenAIError as e:
                print(f"API call failed: {e}")
                time.sleep(5)
    return outputs


@backoff.on_exception(backoff.expo, openai.error.OpenAIError, max_tries=5, max_time=10)
def completions_with_backoff(**kwargs):
    """Call OpenAI ChatCompletion API with exponential backoff retry logic."""
    return openai.ChatCompletion.create(**kwargs)

def gpt(prompt, model="Qwen2.5-7B-Instruct-1M", temperature=0.2, max_tokens=200, n=1, stop=None) -> list:
    """Main GPT function for generating responses.
    
    Args:
        prompt: The prompt to send to the model
        model: Model name to use (default model)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        n: Number of completions to generate
        stop: Stop sequences
        
    Returns:
        List of [response, model_name] pairs
    """
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)


def chatgpt(messages, model="Qwen2.5-14B-Instruct", temperature=0.2, max_tokens=200, n=1, stop=None) -> list:
    """Chat completion with multi-model agent pool support.
    
    Args:
        messages: List of message dictionaries
        model: Default model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        n: Number of completions to generate
        stop: Stop sequences
        
    Returns:
        List of [response, model_name] pairs
    """
    global completion_tokens, prompt_tokens
    from run import llm_manager
    outputs = []

    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        for i in range(cnt):
            try:
                selected_model = llm_manager.call_llm()
                print(f"Using model: {selected_model}")
                
                res = completions_with_backoff(
                    model=selected_model, 
                    messages=messages, 
                    temperature=temperature,
                    max_tokens=max_tokens, 
                    n=1, 
                    stop=stop
                )

                temp_list = []
                temp_list.append([choice["message"]["content"] for choice in res["choices"]])
                temp_list.append(selected_model)
                outputs.append(temp_list)

                text_temp = [choice["message"]["content"].strip() for choice in res["choices"]]
                logging.info(f"Model: {selected_model}, Response: {text_temp}")

                completion_tokens += res["usage"]["completion_tokens"]
                prompt_tokens += res["usage"]["prompt_tokens"]
                logging.info(f"Completion tokens: {completion_tokens}, Prompt tokens: {prompt_tokens}")

            except openai.error.OpenAIError as e:
                print(f"API call failed: {e}")
                time.sleep(5)

    logging.info(f"LLM manager stats: {llm_manager.llm_stats}")
    print(f"LLM manager status: {llm_manager.llm_stats}")
    return outputs
    
def gpt_usage():
    """Get current token usage statistics.
    
    Returns:
        Dictionary with completion_tokens and prompt_tokens counts
    """
    global completion_tokens, prompt_tokens
    return {
        "completion_tokens": completion_tokens, 
        "prompt_tokens": prompt_tokens
    }