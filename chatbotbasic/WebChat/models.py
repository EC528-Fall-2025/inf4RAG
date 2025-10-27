import os
import openai
import anthropic
import huggingface_hub
import tiktoken # Tokenizer for OpenAI GPT models
import sentencepiece # Tokenizer for LLaMA 2 model
from openai import OpenAI

import time

MAX_TOKENS = 1000  # Max number of tokens that each model should generate

class Model:
    """
    Common interface for all chat model API providers
    """
    '''
    def __init__(self, config):
        self.config = config
        self.client = OpenAI(
            base_url=config.get("base_url", "http://127.0.0.1:8000/v1"),
            api_key=config.get("api_key", "ec528")
        )
        self.model_name = self.client.models.list().data[0].id
        print(f"Using model: {self.model_name}")
    '''
    def __init__(self, config):
        self.config = config
        try:
            self.client = OpenAI(
                base_url=config.get("base_url", "http://127.0.0.1:8000/v1"),
                api_key=config.get("api_key", "ec528")
            )
            self.model_name = self.client.models.list().data[0].id
            print(f"Using model: {self.model_name}")
            self.connected = True
        except Exception as e:
            print(f"Warning: Could not connect to vLLM: {e}")
            print("Using mock responses. Please start vLLM service on OpenStack.")
            self.model_name = "mock-model"
            self.connected = False
            self.client = None

    def _generate_mock_response(self, query):
        """
        Generate a mock streaming response when vLLM is not available.
        """
        mock_text = f"""This is a mock response. Your question was: "{query}"

    NOTE: Currently unable to connect to vLLM service on OpenStack.

    To enable real responses:
    1. Launch an instance on OpenStack
    2. Mount the 'All-Models' volume
    3. Start vLLM service with: vllm serve /data/Phi-3-mini-4k-instruct --api-key=ec528
    4. Ensure the floating IP is accessible

    Check README.md for detailed instructions."""
        
        # Simulate streaming by yielding chunks
        class MockStream:
            def __init__(self, text):
                self.text = text
                self.words = text.split()
                self.index = 0
            
            def __iter__(self):
                return self
            
            def __next__(self):
                if self.index >= len(self.words):
                    raise StopIteration
                
                word = self.words[self.index]
                self.index += 1
                
                class MockCompletion:
                    class choices:
                        class delta:
                            content = word + " "
                
                return MockCompletion()
        
        return MockStream(mock_text)
    
    '''
    def generate(self, system_message, new_user_message, history=[], temperature=1):
        messages = [{"role": "system", "content": system_message}]

        for user_message, assistant_response in history:
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": assistant_response})

        messages.append({"role": "user", "content": new_user_message})

        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=MAX_TOKENS,
            stream=True,
        )

        return stream
    '''

    def generate(self, system_message, new_user_message, history=[], temperature=1):
        # If not connected, return mock stream
        if not self.connected or self.client is None:
            return self._generate_mock_response(new_user_message)
        
        messages = [{"role": "system", "content": system_message}]

        for user_message, assistant_response in history:
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": assistant_response})

        messages.append({"role": "user", "content": new_user_message})

        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=MAX_TOKENS,
                stream=True,
            )
            return stream
        except Exception as e:
            print(f"Error calling vLLM: {e}")
            return self._generate_mock_response(new_user_message)

    '''
    def parse_completion(self, completion):
        # ✅ works with ChatCompletionChunk
        delta = completion.choices[0].delta
        if delta.content:
            return delta.content
        return None
    '''

    def parse_completion(self, completion):
        # ✅ works with ChatCompletionChunk
        try:
            delta = completion.choices[0].delta
            if delta.content:
                return delta.content
        except AttributeError:
            return completion.choices.delta.content
        return None

class OpenAIModel(Model):
    """
    Interface for OpenAI's GPT models
    """
    '''
    def generate(self, system_message, new_user_message, history=[], temperature=1):
        messages = [{"role": "system", "content": system_message}]

        for user_message, assistant_response in history:
            if user_message:
                messages.append({"role": "user", "content": str(user_message)})
            if assistant_response:
                messages.append({"role": "assistant", "content": str(assistant_response)})

        messages.append({"role": "user", "content": str(new_user_message)})

        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=MAX_TOKENS,
            stream=True,
        )

        return stream
    
    def parse_completion(self, completion):
        delta = completion.choices[0].delta
        if delta.content:
            return delta.content
        return None
    '''
    def generate(self, system_message, new_user_message, history=[], temperature=1):
        # If not connected, return mock stream
        if not self.connected or self.client is None:
            return self._generate_mock_response(new_user_message)
        
        messages = [{"role": "system", "content": system_message}]

        for user_message, assistant_response in history:
            if user_message:
                messages.append({"role": "user", "content": str(user_message)})
            if assistant_response:
                messages.append({"role": "assistant", "content": str(assistant_response)})

        messages.append({"role": "user", "content": new_user_message})

        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=MAX_TOKENS,
                stream=True,
            )
            return stream
        except Exception as e:
            print(f"Error calling vLLM: {e}")
            return self._generate_mock_response(new_user_message)
    
    def parse_completion(self, completion):
        # ✅ works with ChatCompletionChunk
        try:
            delta = completion.choices[0].delta
            if delta.content:
                return delta.content
        except AttributeError:
            return completion.choices.delta.content
        return None