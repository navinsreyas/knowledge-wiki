"""Export fine-tuned wiki model to GGUF for Ollama deployment."""

from unsloth import FastLanguageModel


def run_export():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="./outputs/wiki-finetuned",
        max_seq_length=2048,
        load_in_4bit=True,
    )

    model.save_pretrained_gguf(
        "outputs/wiki-finetuned-gguf",
        tokenizer,
        quantization_method="q4_k_m",
    )

    print("Exported to outputs/wiki-finetuned-gguf/")
    print("Next steps:")
    print("  echo FROM ./outputs/wiki-finetuned-gguf/model.gguf > Modelfile")
    print("  ollama create wiki-qwen -f Modelfile")
    print("  ollama run wiki-qwen 'What are the tradeoffs between LangGraph and AutoGen?'")


if __name__ == "__main__":
    run_export()
