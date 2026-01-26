"""
Local LLM inference optimized for Mac (no quantization).
"""
from typing import Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    BitsAndBytesConfig
)
from threading import Thread
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from src.config import settings
from src.core.observability import get_logger

logger = get_logger(__name__)

# Thread pool for async inference
_inference_executor = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="llm_inference")


class LocalLLM:
    """Local LLM optimized for Mac with KV cache."""

    _instance = None
    _model = None
    _tokenizer = None
    _model_device = None

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize local LLM."""
        if self._model is None:
            self._load_model()

    def _load_model(self):
        """Load model with optimized device and precision."""
        device = settings.local_llm_device_detected
        torch_device = torch.device(device)
        model_id = settings.local_llm_model

        logger.info(
            "loading_local_llm",
            model=model_id,
            device=device,
            quantization="auto"
        )

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )

        # Set pad token if not already set
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Optimization Strategy:
        # 1. If MPS/CUDA is available, use float16 for high speed.
        # 2. If only CPU is available, use 8-bit quantization for memory efficiency.

        try:
            if device in ["mps", "cuda"]:
                logger.info(f"using_gpu_acceleration",
                            device=device, dtype="float16")
                # Use 'dtype' instead of deprecated 'torch_dtype'
                # Use 'device_map="auto"' to let Transformers handle robust device placement
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    use_cache=True,
                    attn_implementation="sdpa"
                )
                self._model_device = device
            else:
                logger.info("using_cpu_quantization", dtype="int8")
                # Use standard BitsAndBytesConfig flags: load_in_8bit or load_in_4bit.
                # The prior code used a non-standard flag `load_in_8bit_int8_cpu` which
                # is not accepted by the transformers BitsAndBytesConfig API.
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
                # Force device_map to CPU for quantized weights to avoid accidental GPU dispatch
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=quantization_config,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    device_map={"": "cpu"},
                    use_cache=True
                )
                self._model_device = "cpu"

            self._model.eval()
            logger.info("llm_loaded_successfully", device=device)

        except Exception as e:
            logger.error("llm_loading_failed", error=str(e))
            # Fallback to plain CPU float32 if everything else fails
            logger.info("falling_back_to_cpu_float32")
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_cache=True,
                device_map={"": "cpu"}
            )
            self._model.eval()
            self._model_device = "cpu"

    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        use_template: bool = True,
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate text.

        Args:
            prompt: Input prompt
            max_length: Max tokens to generate
            use_template: Whether to apply chat template
            do_sample: Whether to use sampling (greedy if False)
            temperature: Generation temperature
        """
        start_time = time.time()
        
        # Determine sampling and temperature
        temp = temperature if temperature is not None else settings.local_llm_temperature
        # If temperature is 0, default to greedy decoding
        should_sample = do_sample if do_sample is not None else (temp > 0)
        
        logger.info("generation_started", prompt_len=len(prompt), use_template=use_template, do_sample=should_sample)

        # Format prompt
        if use_template:
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ]

            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            text = prompt

        # Tokenize and move to model's device
        inputs = self._tokenizer(
            text, return_tensors="pt").to(self._model_device)

        # Generate with KV cache
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_length or settings.local_llm_max_length,
                temperature=temp if should_sample else None,
                top_p=settings.local_llm_top_p if should_sample else None,
                top_k=settings.local_llm_top_k if should_sample else None,
                do_sample=should_sample,
                use_cache=True,  # KV cache
                pad_token_id=self._tokenizer.eos_token_id,
                repetition_penalty=settings.local_llm_repetition_penalty
            )

        # Decode
        result = self._tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        elapsed = time.time() - start_time
        logger.info("generation_completed",
                    elapsed_seconds=elapsed, result_len=len(result))

        return result

    async def agenerate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        use_template: bool = True,
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Async generate using thread pool to avoid blocking.
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _inference_executor,
            self.generate,
            prompt,
            max_length,
            use_template,
            do_sample,
            temperature
        )
        return result

    async def astream(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        use_template: bool = True,
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
    ):
        """
        Stream generation token by token asynchronously.
        """
        logger.info("streaming_generation", prompt_len=len(prompt), use_template=use_template)
        # Determine sampling and temperature
        temp = temperature if temperature is not None else settings.local_llm_temperature
        should_sample = do_sample if do_sample is not None else (temp > 0)
        
        # Format prompt
        if use_template:
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ]

            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            text = prompt

        # Tokenize and move to model's device
        inputs = self._tokenizer(
            text, return_tensors="pt").to(self._model_device)

        # Create streamer
        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # Generation kwargs
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_length or settings.local_llm_max_length,
            temperature=temp if should_sample else None,
            top_p=settings.local_llm_top_p if should_sample else None,
            top_k=settings.local_llm_top_k if should_sample else None,
            do_sample=should_sample,
            streamer=streamer,
            use_cache=True,
            pad_token_id=self._tokenizer.eos_token_id,
            repetition_penalty=settings.local_llm_repetition_penalty
        )

        # Start generation in thread
        thread = Thread(target=self._model.generate,
                        kwargs=gen_kwargs, daemon=True)
        thread.start()

        # Stream tokens
        try:
            for token in streamer:
                yield token
                await asyncio.sleep(0)
        finally:
            thread.join(timeout=5)


# Global instance
def get_local_llm() -> LocalLLM:
    """Get the global LLM instance."""
    return LocalLLM()
