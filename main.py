"""Train a GPT-2 small model from scratch on The Lord of the Rings and log to W&B.

--smoke-test checks the pipeline locally (W&B disabled), --test is a short cluster
check with W&B enabled, a few logged batches and no checkpoint.
"""

import argparse
import os
import shutil
import urllib.request
from dataclasses import replace
from pathlib import Path

from infm_sota2.data import TextSplits
from infm_sota2.env import Environment
from infm_sota2.model import GPT2_SMALL_CONFIG, GPTModel, build_model
from infm_sota2.training import TrainConfig, Trainer

BOOK_URL = (
    "https://dn721603.ca.archive.org/0/items/"
    "tolkien-j.-the-lord-of-the-rings-harper-collins-ebooks-2010/"
    "Tolkien-J.-The-lord-of-the-rings-HarperCollins-ebooks-2010_djvu.txt"
)
BOOK_FILE = "lotr_djvu.txt"
BOOK_MARKERS = {
    "start_line": "PROLOGUE",
    "end_line": "APPENDIX A",
    "book_title": "THE LORD OF THE RINGS",
}
SEQ_LEN = 256
BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 4e-4
SEED = 123
GROUP = "gpt2-scratch-lotr"
CHECKPOINT = "gpt2-lotr-scratch-best.pth"


def download_book(data_dir: Path) -> Path:
    """Cache the OCR text under data/; skip the download if it is already there."""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / BOOK_FILE
    if path.exists():
        return path
    partial = path.with_suffix(".txt.part")
    with (
        urllib.request.urlopen(BOOK_URL, timeout=60) as response,
        partial.open("wb") as file,
    ):
        shutil.copyfileobj(response, file)
    partial.replace(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--smoke-test",
        action="store_true",
        help="local pipeline check: tiny model, short text, W&B disabled",
    )
    mode.add_argument(
        "--test",
        action="store_true",
        help="cluster check: tiny model, logged batches, W&B on, no checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    tiny = args.smoke_test or args.test
    env = Environment.detect(smoke_test=args.smoke_test)
    # Trainer passes mode to wandb.init(), which overrides WANDB_MODE from the shell.
    if os.environ.get("WANDB_MODE") == "offline":
        env = replace(env, wandb_mode="offline")

    text_path = download_book(env.data_dir)
    seq_len = 32 if tiny else SEQ_LEN
    max_chars = {"train": 30_000, "val": 6_000, "test": 6_000} if tiny else None
    data = TextSplits.from_book(
        env.data_dir,
        file=text_path.name,
        **BOOK_MARKERS,
        train_ratio=0.8,
        val_ratio=0.1,
        max_chars=max_chars,
        seq_len=seq_len,
    )
    print(f"Text: {text_path}")
    print(data.summary())
    print(data.split_summary(BATCH_SIZE))

    model_config = dict(GPT2_SMALL_CONFIG)
    if tiny:
        model_config.update(context_length=seq_len, emb_dim=32, n_heads=2, n_layers=2)
    if args.test:
        eval_every = 1
    elif args.smoke_test:
        eval_every = 5
    else:
        eval_every = 100
    train_config = TrainConfig(
        weight_decay=0.1,
        batch_size=BATCH_SIZE,
        eval_every=eval_every,
        eval_batches=2 if tiny else None,
        grad_clip=1.0,
        use_amp=env.device.type == "cuda",
        seed=SEED,
    )

    model = build_model(GPTModel, model_config, SEED, env.device)
    print(
        f"Model: {sum(p.numel() for p in model.parameters()):,} parameters "
        f"on {env.device}"
    )

    trainer = Trainer(
        env,
        wandb_config={
            "model": model_config,
            "data": {
                "file": text_path.name,
                **BOOK_MARKERS,
                "train_ratio": 0.8,
                "val_ratio": 0.1,
                "max_chars": max_chars,
                "seq_len": seq_len,
            },
            "train": {
                "batch_size": BATCH_SIZE,
                "weight_decay": train_config.weight_decay,
                "grad_clip": train_config.grad_clip,
                "eval_every": eval_every,
                "seed": SEED,
                "mixed_precision": train_config.use_amp,
            },
            "environment": env.describe(),
        },
    )
    if env.wandb_mode == "online":
        env.login_wandb()

    suffix = ""
    if args.smoke_test:
        suffix = "-smoke"
    elif args.test:
        suffix = "-test"
    trainer.fit(
        model,
        data,
        train_config,
        learning_rate=args.lr,
        epochs=1 if tiny else args.epochs,
        name=f"gpt2-scratch-lr{args.lr:g}{suffix}",
        group=GROUP,
        checkpoint_name=None if tiny else CHECKPOINT,
    )


if __name__ == "__main__":
    main()
