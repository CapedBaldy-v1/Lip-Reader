"""
Lightweight NLP utilities for translation and summarization.

Designed for low-resource Windows laptops; defaults to CPU inference.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from backend_manager import get_backend, get_device
from logging_utils import setup_logging, log_exception


LOGGER = setup_logging("nlp_utils")


@dataclass
class TranslationOption:
    code: str
    name: str
    model: str


TRANSLATION_OPTIONS: Dict[str, TranslationOption] = {
    "es": TranslationOption("es", "Spanish", "Helsinki-NLP/opus-mt-en-es"),
    "fr": TranslationOption("fr", "French", "Helsinki-NLP/opus-mt-en-fr"),
    "de": TranslationOption("de", "German", "Helsinki-NLP/opus-mt-en-de"),
    "ar": TranslationOption("ar", "Arabic", "Helsinki-NLP/opus-mt-en-ar"),
    "zh": TranslationOption("zh", "Chinese", "Helsinki-NLP/opus-mt-en-zh"),
    "it": TranslationOption("it", "Italian", "Helsinki-NLP/opus-mt-en-it")
}

DEFAULT_SUMMARY_MODEL = os.getenv("SWIN_VALLR_SUMMARY_MODEL", "google/flan-t5-small")


def get_translation_options() -> List[TranslationOption]:
    return list(TRANSLATION_OPTIONS.values())


def _select_device() -> Tuple[torch.device, torch.dtype]:
    preferred = os.getenv("SWIN_VALLR_NLP_DEVICE", "").strip().lower()
    backend = get_backend()
    if preferred == "cpu" or backend == "directml":
        return torch.device("cpu"), torch.float32
    if backend in ("cuda", "rocm"):
        device = get_device()
        return device, torch.float16
    return torch.device("cpu"), torch.float32


@lru_cache(maxsize=8)
def _load_seq2seq(model_name: str, device_type: str, dtype_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
    dtype = torch.float16 if dtype_name == "float16" else torch.float32
    device = torch.device(device_type)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return tokenizer, model, device


def translate_text(
    text: str,
    target_lang: str,
    max_new_tokens: int = 128
) -> str:
    if not text or not target_lang:
        return text
    option = TRANSLATION_OPTIONS.get(target_lang)
    if option is None:
        LOGGER.warning("Unknown translation target: %s", target_lang)
        return text
    try:
        device, dtype = _select_device()
        tokenizer, model, device = _load_seq2seq(
            option.model,
            device.type,
            "float16" if dtype == torch.float16 else "float32"
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return translated or text
    except Exception:
        log_exception(LOGGER, "Translation failed.")
        return text


def summarize_text(
    text: str,
    max_new_tokens: int = 160,
    max_input_tokens: int = 512
) -> str:
    if not text:
        return ""
    try:
        device, dtype = _select_device()
        tokenizer, model, device = _load_seq2seq(
            DEFAULT_SUMMARY_MODEL,
            device.type,
            "float16" if dtype == torch.float16 else "float32"
        )
        model_name = DEFAULT_SUMMARY_MODEL.lower()
        prompt = text
        if "t5" in model_name:
            prompt = f"summarize: {text}"

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return summary
    except Exception:
        log_exception(LOGGER, "Summary failed.")
        return ""
