import os
os.environ["UNSLOTH_DISABLE_FUSED_CE"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

# If CUDA OOM: set per_device_train_batch_size=1, gradient_accumulation_steps=8
# Watch WandB: eval_loss should plateau, not rise. If it rises, overfitting.
# load_best_model_at_end=True saves the best checkpoint automatically.

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset


def format_pair(example: dict) -> dict:
    return {
        "text": (
            f"<|im_start|>user\n{example['question']}<|im_end|>\n"
            f"<|im_start|>assistant\n{example['answer']}<|im_end|>"
        )
    }


def run_finetune():
    torch.cuda.empty_cache()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        max_seq_length = 512,
        load_in_4bit   = True,
        dtype          = None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r              = 8,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha     = 8,
        lora_dropout   = 0,
        bias           = "none",
        use_gradient_checkpointing = "unsloth",
    )

    train_dataset = load_dataset("json", data_files="finetune/train.json")["train"]
    eval_dataset  = load_dataset("json", data_files="finetune/val.json")["train"]

    train_dataset = train_dataset.map(format_pair)
    eval_dataset  = eval_dataset.map(format_pair)

    args = SFTConfig(
        output_dir                  = "./outputs",
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        num_train_epochs            = 3,
        learning_rate               = 2e-4,
        bf16                        = True,
        fp16                        = False,
        optim                       = "adamw_8bit",
        eval_strategy               = "steps",
        eval_steps                  = 50,
        save_strategy               = "steps",
        save_steps                  = 50,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        report_to                   = "wandb",
        dataset_text_field          = "text",
        max_seq_length              = 512,
        dataloader_num_workers      = 0,
        gradient_checkpointing      = True,
    )

    trainer = SFTTrainer(
        model         = model,
        tokenizer     = tokenizer,
        train_dataset = train_dataset,
        eval_dataset  = eval_dataset,
        args          = args,
    )

    torch.cuda.empty_cache()
    trainer.train()
    model.save_pretrained("./outputs/wiki-finetuned")
    tokenizer.save_pretrained("./outputs/wiki-finetuned")
    print("Training complete. Best checkpoint saved to ./outputs/wiki-finetuned")


if __name__ == "__main__":
    run_finetune()
