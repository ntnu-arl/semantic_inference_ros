from typing import Dict, Tuple
from openai import OpenAI
import os

from semantic_inference_python.client.config import OpenAIClientConfig


class OpenAIClient:
    """Client for OpenAI API to generate navigation prompts."""

    def __init__(
        self,
        config: OpenAIClientConfig,
        system_prompt=None,
        use_completion: bool = True,
    ) -> None:
        """Initialize OpenAI client with model, max tokens, and system prompt.
        :param config: Client configuration containing API key, model, and max tokens.
        :param system_prompt: Optional system prompt to guide the model's responses.
        :param use_completion: Whether to use the completion endpoint (default is True).
        """
        self.config = config
        self._system_prompt = system_prompt
        self._use_completion = use_completion
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

    def generate_response(self, prompt: str, log: bool = False) -> Tuple[Dict, bool]:
        """Generate a response from the OpenAI model.
        :param prompt: The input prompt to send to the model.
        :return: A tuple containing the response dictionary and a boolean indicating success."""
        try:

            if self._use_completion:
                messages = [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": prompt},
                ]
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    # temperature=0
                )
                if log:

                    print(f"[OpenAIClient] System prompt: {self._system_prompt}")
                    print(f"[OpenAIClient] User prompt: {prompt}")
                    print(
                        f"[OpenAIClient] Response: {response.choices[0].message.content}"
                    )
                if response.choices and response.choices[0].message:
                    return response.choices[0].message.content, True
                else:
                    return {"error": "No valid response from OpenAI"}, False
            else:
                response = self.client.responses.create(
                    model=self.config.model,
                    input=prompt,
                    instructions=self._system_prompt,
                    max_output_tokens=self.config.max_tokens,
                )
                if response.error is not None:
                    return response.error.message, False
                return response.output_text, True
        except Exception as e:
            print(f"[OpenAIClient] Error generating response: {e}")
            return {"error": str(e)}, False
