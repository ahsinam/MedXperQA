from openai import OpenAI
from typing import List
from model.base_agent import LLMAgent
import base64
import traceback
import time
import os

# Extra providers added for this reproduction: model name -> (API address, name of the key variable)
EXTRA_MODELS = {
    "meta-llama/llama-3.3-70b-instruct": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "llama-3.3-70b-versatile": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openai/gpt-oss-120b": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openai/gpt-oss-20b": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
}

def encode_image(image_path):
    if image_path:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    else:
        return "No image inputs"

class APIAgent(LLMAgent):
    def __init__(self, model_name, temperature=0) -> None:
        super().__init__(model_name, temperature)

        if "gpt-oss" in model_name or "o3" in model_name or "o1" in model_name or "qwq" in model_name.lower() or "qvq" in model_name.lower() or "reasoner" in model_name.lower():
            self.max_tokens = 8192
        else:
            self.max_tokens = 2048

        if os.environ.get("VLLM_URL"):
            # A vLLM server running on this machine (e.g. Colab) serves the model.
            # It answers many questions at once, so use --num-threads 32 with it.
            print(f"Using local vLLM server at {os.environ['VLLM_URL']}")
            self.local_server = True
            self.client = OpenAI(api_key="EMPTY", base_url=os.environ["VLLM_URL"])
        elif model_name in EXTRA_MODELS:
            base_url, key_name = EXTRA_MODELS[model_name]
            print(f"Using {base_url}")
            self.client = OpenAI(api_key=os.environ[key_name], base_url=base_url)
        elif model_name in ["deepseek-chat", "deepseek-reasoner"]:
            print("DeepSeek")
            self.client = OpenAI(api_key="[API_KEY]", base_url="https://api.deepseek.com/v1")
        else:
            print("OpenAI")
            self.client = OpenAI(api_key="[API_KEY]", base_url="[BASE_URL]")

    def get_response(self, messages: List[dict]):
        if getattr(self, "local_server", False) or self.model_name in EXTRA_MODELS:
            return self.get_extra_response(messages)
        if ("o3" in self.model_name) or ("o1" in self.model_name) or ("deepseek-reasoner" in self.model_name):
            messages = [m for m in messages if m["role"] != "system"]
            for _ in range(10):
                try:
                    completion = self.client.chat.completions.create(
                        messages=messages, model=self.model_name,
                        max_completion_tokens=self.max_tokens, seed=0)
                    response = completion.choices[0].message.content
                    break
                except Exception as e:
                    print(e); time.sleep(1); response = "No answer provided."
        else:
            for _ in range(10):
                try:
                    completion = self.client.chat.completions.create(
                        messages=messages, model=self.model_name, temperature=self.temperature,
                        max_tokens=self.max_tokens, logprobs=True, seed=0)
                    response = completion.choices[0].message.content
                    break
                except Exception as e:
                    print(e); time.sleep(1); response = "No answer provided."
        try:
            log_probs = [t.logprob for t in completion.choices[0].logprobs.content]
        except Exception:
            log_probs = []
        return response, log_probs

    def get_extra_response(self, messages: List[dict]):
        # Plain-string messages and no logprobs (Groq rejects both).
        clean = []
        for m in messages:
            content = m["content"]
            if isinstance(content, list):
                content = "\n".join(c["text"] for c in content if c.get("type") == "text")
            clean.append({"role": m["role"], "content": content})
        wait = 5
        for _ in range(30):
            try:
                completion = self.client.chat.completions.create(
                    messages=clean, model=self.model_name, temperature=self.temperature,
                    max_tokens=self.max_tokens, seed=0)
                return completion.choices[0].message.content or "", []
            except Exception as e:
                # Usually the free-tier rate limit: wait longer each time.
                print(f"API error, retrying in {wait}s: {e}")
                time.sleep(wait)
                wait = min(wait * 2, 120)
        raise RuntimeError("API kept failing; stopping instead of recording a fake wrong answer.")

    def image_content(self, img_path: str) -> dict:
        img_path = img_path.strip()
        if img_path.startswith("http"):
            return {"type": "image_url", "image_url": {"url": img_path}}
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img_path)}"}}

# from openai import OpenAI
# from typing import List
# from model.base_agent import LLMAgent
# import base64
# import traceback
# import time
# import os

# # Extra providers added for this reproduction: model name -> (API address, name of the key variable)
# EXTRA_MODELS = {
#     "meta-llama/llama-3.3-70b-instruct": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
#     "llama-3.3-70b-versatile": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
#     "openai/gpt-oss-120b": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
#     "openai/gpt-oss-20b": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
# }

# def encode_image(image_path):
#     if image_path:
#         with open(image_path, "rb") as image_file:
#             return base64.b64encode(image_file.read()).decode("utf-8")
#     else:
#         return "No image inputs"

# class APIAgent(LLMAgent):
#     def __init__(self, model_name, temperature=0) -> None:
#         super().__init__(model_name, temperature)

#         if "gpt-oss" in model_name or "o3" in model_name or "o1" in model_name or "qwq" in model_name.lower() or "qvq" in model_name.lower() or "reasoner" in model_name.lower():
#             self.max_tokens = 8192
#         else:
#             self.max_tokens = 2048

#         if model_name in EXTRA_MODELS:
#             base_url, key_name = EXTRA_MODELS[model_name]
#             print(f"Using {base_url}")
#             self.client = OpenAI(api_key=os.environ[key_name], base_url=base_url)
#         elif model_name in ["deepseek-chat", "deepseek-reasoner"]:
#             print("DeepSeek")
#             self.client = OpenAI(api_key="[API_KEY]", base_url="https://api.deepseek.com/v1")
#         else:
#             print("OpenAI")
#             self.client = OpenAI(api_key="[API_KEY]", base_url="[BASE_URL]")

#     def get_response(self, messages: List[dict]):
#         if self.model_name in EXTRA_MODELS:
#             return self.get_extra_response(messages)
#         if ("o3" in self.model_name) or ("o1" in self.model_name) or ("deepseek-reasoner" in self.model_name):
#             messages = [m for m in messages if m["role"] != "system"]
#             for _ in range(10):
#                 try:
#                     completion = self.client.chat.completions.create(
#                         messages=messages, model=self.model_name,
#                         max_completion_tokens=self.max_tokens, seed=0)
#                     response = completion.choices[0].message.content
#                     break
#                 except Exception as e:
#                     print(e); time.sleep(1); response = "No answer provided."
#         else:
#             for _ in range(10):
#                 try:
#                     completion = self.client.chat.completions.create(
#                         messages=messages, model=self.model_name, temperature=self.temperature,
#                         max_tokens=self.max_tokens, logprobs=True, seed=0)
#                     response = completion.choices[0].message.content
#                     break
#                 except Exception as e:
#                     print(e); time.sleep(1); response = "No answer provided."
#         try:
#             log_probs = [t.logprob for t in completion.choices[0].logprobs.content]
#         except Exception:
#             log_probs = []
#         return response, log_probs

#     def get_extra_response(self, messages: List[dict]):
#         # Plain-string messages and no logprobs (Groq rejects both).
#         clean = []
#         for m in messages:
#             content = m["content"]
#             if isinstance(content, list):
#                 content = "\n".join(c["text"] for c in content if c.get("type") == "text")
#             clean.append({"role": m["role"], "content": content})
#         wait = 5
#         for _ in range(30):
#             try:
#                 completion = self.client.chat.completions.create(
#                     messages=clean, model=self.model_name, temperature=self.temperature,
#                     max_tokens=self.max_tokens, seed=0)
#                 return completion.choices[0].message.content or "", []
#             except Exception as e:
#                 # Usually the free-tier rate limit: wait longer each time.
#                 print(f"API error, retrying in {wait}s: {e}")
#                 time.sleep(wait)
#                 wait = min(wait * 2, 120)
#         raise RuntimeError("API kept failing; stopping instead of recording a fake wrong answer.")

#     def image_content(self, img_path: str) -> dict:
#         img_path = img_path.strip()
#         if img_path.startswith("http"):
#             return {"type": "image_url", "image_url": {"url": img_path}}
#         return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img_path)}"}}
