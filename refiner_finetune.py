"""
LoRA fine-tuning for the refiner.

Supports:
- phoneme mode: map phoneme sequences to text (English-focused)
- roman mode: map romanized text to target language text (paper-12 inspired)
"""

import argparse
import math
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

from data_pipeline import text_to_phonemes
from model_architecture import LinguisticRefiner
from logging_utils import setup_logging, log_system_info, log_exception
from artifact_utils import (
    get_artifact_root,
    get_useful_dir,
    save_json,
    save_text,
    save_csv,
    save_line_plot
)


LOGGER = setup_logging("refiner_finetune")


def normalize_text(text: str, mode: str = "phoneme") -> str:
    """Normalize input text based on training mode."""
    text = text.replace("\ufeff", " ").strip()
    if mode == "roman":
        text = re.sub(r"\s+", " ", text).strip()
        return text
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def collect_text_files(paths: Sequence[str]) -> List[Path]:
    """Collect text files from file or directory inputs."""
    files: List[Path] = []
    for entry in paths:
        path = Path(entry)
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.txt")))
    return files


def load_text_lines(
    paths: Sequence[str],
    max_samples: Optional[int] = None,
    mode: str = "phoneme"
) -> List[str]:
    """Load and normalize lines from text sources."""
    lines: List[str] = []
    files = collect_text_files(paths)
    for path in tqdm(files, desc="Loading corpus", unit="file"):
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            LOGGER.warning("Failed to read file: %s", path)
            continue
        for line in raw.splitlines():
            cleaned = normalize_text(line, mode=mode)
            if cleaned:
                lines.append(cleaned)
                if max_samples and len(lines) >= max_samples:
                    return lines
    return lines


def build_prompt(phonemes: List[str]) -> str:
    """Build a prompt consistent with the inference prompt."""
    refiner = LinguisticRefiner()
    hint = refiner.phonemes_to_grapheme_hint(phonemes)
    phoneme_str = " ".join(phonemes)
    return (
        "You are an expert lip-reading assistant. Convert the following phoneme sequence "
        "to proper English text.\n\n"
        f"Phoneme sequence: {phoneme_str}\n"
        f"Approximate sounds: {hint}\n\n"
        "Output only the English text, nothing else:"
    )

def build_prompt_roman(roman_text: str, language_hint: Optional[str]) -> str:
    """Build a romanized prompt consistent with the inference prompt."""
    target = language_hint.strip() if language_hint and language_hint.strip() else "English"
    return (
        "You are an expert language assistant. Convert the following romanized speech "
        f"into proper {target} text.\n\n"
        f"Roman text: {roman_text}\n\n"
        f"Output only the {target} text, nothing else:"
    )


def build_samples(
    lines: List[str],
    max_samples: Optional[int] = None,
    min_words: int = 1,
    mode: str = "phoneme",
    language_hint: Optional[str] = None
) -> List[Dict[str, str]]:
    """Convert text lines into prompt/target training samples."""
    samples: List[Dict[str, str]] = []
    for line in tqdm(lines, desc="Building samples", unit="line"):
        if len(line.split()) < min_words:
            continue
        if mode == "roman":
            roman_text = LinguisticRefiner.romanize_text(line)
            if not roman_text:
                continue
            prompt = build_prompt_roman(roman_text, language_hint)
            samples.append({"prompt": prompt, "target": line, "roman": roman_text})
        else:
            phonemes = text_to_phonemes(line)
            if not phonemes:
                continue
            prompt = build_prompt(phonemes)
            samples.append({"prompt": prompt, "target": line})
        if max_samples and len(samples) >= max_samples:
            break
    return samples


class PromptDataset(Dataset):
    """Dataset of prompt/target pairs for LoRA fine-tuning."""

    def __init__(self, samples: List[Dict[str, str]], tokenizer, max_length: int):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        sample = self.samples[idx]
        prompt = sample["prompt"]
        target = sample["target"]

        full_text = f"{prompt} {target}"
        full_enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True
        )
        prompt_enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True
        )

        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]
        prompt_len = min(len(prompt_enc["input_ids"]), len(input_ids))

        labels = input_ids.copy()
        for i in range(prompt_len):
            labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def pad_batch(batch: List[Dict[str, List[int]]], pad_id: int) -> Dict[str, torch.Tensor]:
    """Pad batch to the maximum sequence length."""
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    attention_masks = []
    labels = []

    for item in batch:
        ids = item["input_ids"]
        mask = item["attention_mask"]
        lab = item["labels"]

        pad_amount = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad_amount)
        attention_masks.append(mask + [0] * pad_amount)
        labels.append(lab + [-100] * pad_amount)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune refiner with LoRA on phoneme->text or roman->text pairs."
    )
    parser.add_argument("--mode", choices=["phoneme", "roman"], default="phoneme",
                        help="Training mode: phoneme (default) or roman.")
    parser.add_argument("--language_hint", type=str, default="",
                        help="Target language label for roman mode (e.g., English, Spanish).")
    parser.add_argument("--corpus", action="append", default=[],
                        help="Text file or directory to use as corpus (repeatable).")
    parser.add_argument("--labels_dir", type=str,
                        help="Optional labels directory (.txt files) to include.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the LoRA adapter and artifacts.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B-Instruct",
                        help="Base model to fine-tune.")
    parser.add_argument("--max_samples", type=int, default=50000,
                        help="Maximum number of samples to use.")
    parser.add_argument("--min_words", type=int, default=1,
                        help="Minimum number of words per sample.")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Max token length for training.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size.")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate.")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Warmup steps.")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation steps.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout.")
    parser.add_argument("--fp16", action="store_true",
                        help="Use fp16 on CUDA/ROCm.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_system_info(LOGGER)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    mode = args.mode
    language_hint = args.language_hint.strip() or None

    corpus_paths = list(args.corpus)
    if args.labels_dir:
        corpus_paths.append(args.labels_dir)

    if not corpus_paths:
        raise ValueError("Provide --corpus and/or --labels_dir.")

    LOGGER.info("Loading corpus from %s (mode=%s)", corpus_paths, mode)
    lines = load_text_lines(corpus_paths, max_samples=args.max_samples, mode=mode)
    samples = build_samples(
        lines,
        max_samples=args.max_samples,
        min_words=args.min_words,
        mode=mode,
        language_hint=language_hint
    )

    if not samples:
        raise ValueError("No training samples were generated.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = PromptDataset(samples, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: pad_batch(batch, tokenizer.pad_token_id)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = args.fp16 and device.type in ("cuda", "rocm")
    dtype = torch.float16 if use_fp16 else torch.float32

    LOGGER.info("Loading model %s", args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=True
    )
    model = model.to(device)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = math.ceil(len(dataloader) / args.grad_accum) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )

    step_metrics: List[Tuple[int, float]] = []
    epoch_metrics: List[Tuple[int, float]] = []
    global_step = 0

    LOGGER.info("Starting LoRA fine-tuning.")
    for epoch in range(args.epochs):
        running_loss = 0.0
        steps_in_epoch = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            loss_value = float(loss.item()) * args.grad_accum
            running_loss += loss_value
            steps_in_epoch += 1
            global_step += 1

            step_metrics.append((global_step, loss_value))
            pbar.set_postfix({"loss": running_loss / max(steps_in_epoch, 1)})

        avg_loss = running_loss / max(steps_in_epoch, 1)
        epoch_metrics.append((epoch + 1, avg_loss))

        LOGGER.info("Epoch %s loss: %f", epoch + 1, avg_loss)

        try:
            artifacts = get_useful_dir("refiner_finetune", "loss_curves")
            save_csv(
                artifacts / "step_loss.csv",
                step_metrics,
                headers=["step", "loss"]
            )
            save_csv(
                artifacts / "epoch_loss.csv",
                epoch_metrics,
                headers=["epoch", "loss"]
            )
            save_line_plot(
                artifacts / "step_loss.png",
                {"loss": [loss for _, loss in step_metrics]},
                title="Refiner Step Loss"
            )
            save_line_plot(
                artifacts / "epoch_loss.png",
                {"loss": [loss for _, loss in epoch_metrics]},
                title="Refiner Epoch Loss"
            )
        except Exception:
            log_exception(LOGGER, "Failed to save training artifacts.")

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    LOGGER.info("Saved LoRA adapter to %s", adapter_dir)

    try:
        summary = {
            "model_name": args.model_name,
            "num_samples": len(samples),
            "mode": mode,
            "language_hint": language_hint,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "max_length": args.max_length
        }
        save_json(output_dir / "training_summary.json", summary)
        example_dir = get_useful_dir("refiner_finetune", "samples")
        sample_preview = samples[:5]
        save_json(example_dir / "sample_prompts.json", sample_preview)
        save_text(example_dir / "notes.txt", "Sample prompts and targets used for refiner fine-tuning.")
    except Exception:
        log_exception(LOGGER, "Failed to save training summary.")

    print(f"[Refiner] LoRA adapter saved to: {adapter_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log_exception(LOGGER, "Refiner fine-tuning failed with an unhandled exception.")
        raise
