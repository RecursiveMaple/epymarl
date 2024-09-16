import logging

from zhipuai import ZhipuAI

from utils.logging import suppress_logging_module


class GLMLLM:
    def __init__(self, key, model, top_p, temperature):
        self.set_logging()
        self.client = ZhipuAI(api_key=key)
        self.model = model
        self.top_p = top_p
        self.temperature = temperature
        self.tools = [{"type": "web_search", "web_search": {"enable": False}}]

    def query(self, prompts: tuple[str]) -> str:
        # prompt: (common_text, description)
        messages = [
            {"role": "system", "content": prompts[0]},
            {"role": "user", "content": prompts[1]},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            top_p=self.top_p,
            temperature=self.temperature,
            tools=self.tools,
        )
        return response.choices[0].message.content

    def close(self):
        self.client.close()

    def set_logging(self):
        # suppress logging of zhipuai module and related modules
        suppress_logging_module("zhipuai")
        suppress_logging_module("httpx")
        suppress_logging_module("httpcore")
