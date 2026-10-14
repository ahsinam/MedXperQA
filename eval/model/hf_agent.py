"""Run a Hugging Face model on this machine's own GPU (e.g. Google Colab).

Any --model name that is not in API_MODEL and looks like "organisation/model"
(for example "Qwen/Qwen2.5-3B-Instruct") is downloaded from huggingface.co and
run here, instead of being sent to an online API like Groq.

Optional settings (set them in Colab with os.environ[...] = "..."):
  HF_TOKEN        your Hugging Face token; needed for gated models (Llama, MedGemma)
  HF_4BIT=1       load the model in 4-bit so 7B/8B models fit on a free T4 GPU
                  (needs: pip install bitsandbytes)
  HF_MAX_TOKENS   longest reply the model may write (default 2048)
"""
import os
import threading
from typing import List

from model.base_agent import LLMAgent


class HFAgent(LLMAgent):
    def __init__(self, model_name, temperature=0) -> None:
        super().__init__(model_name, temperature)
        # Imported here so the Groq/API path still works without torch installed.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_tokens = int(os.environ.get("HF_MAX_TOKENS", 2048))
        token = os.environ.get("HF_TOKEN") or None

        if not torch.cuda.is_available():
            print("WARNING: no GPU found. In Colab: Runtime > Change runtime type > T4 GPU. "
                  "On CPU this will be extremely slow.")
        # The free Colab T4 GPU cannot do bfloat16, so use float16 there.
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif torch.cuda.is_available():
            dtype = torch.float16
        else:
            dtype = torch.float32

        load_kwargs = {"torch_dtype": dtype, "device_map": "auto", "token": token}
        if os.environ.get("HF_4BIT") == "1":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4")
            print("Loading in 4-bit")

        print(f"Downloading/loading {model_name} from Hugging Face (first time takes a few minutes)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.eval()
        # One GPU can only answer one question at a time safely.
        self.lock = threading.Lock()
        print("Model ready.")

    @staticmethod
    def _plain_messages(messages: List[dict]) -> List[dict]:
        """The code builds OpenAI-style messages; turn them into plain text ones."""
        clean = []
        for m in messages:
            content = m["content"]
            if isinstance(content, list):
                content = "\n".join(c["text"] for c in content if c.get("type") == "text")
            clean.append({"role": m["role"], "content": content})
        return clean

    def _build_prompt(self, messages: List[dict]):
        try:
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        except Exception:
            # Some models (e.g. older Gemma) refuse a "system" message:
            # fold it into the first user message instead.
            if messages and messages[0]["role"] == "system":
                system, rest = messages[0]["content"], [dict(m) for m in messages[1:]]
                if rest and rest[0]["role"] == "user":
                    rest[0]["content"] = f"{system}\n\n{rest[0]['content']}"
                return self.tokenizer.apply_chat_template(
                    rest, add_generation_prompt=True, return_tensors="pt", return_dict=True)
            raise

    def get_response(self, messages: List[dict]):
        inputs = self._build_prompt(self._plain_messages(messages)).to(self.model.device)
        gen_kwargs = {"max_new_tokens": self.max_tokens,
                      "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id}
        if self.temperature and self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.temperature)
        else:
            gen_kwargs.update(do_sample=False)  # same answer every time (like temperature 0)

        with self.lock, self.torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip(), []

    def image_content(self, img_path: str) -> dict:
        # Text-only for now: image questions are sent without their pictures.
        return {"type": "image_url", "image_url": {"url": img_path}}