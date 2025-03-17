from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ベースモデル unsloth/Llama-3.2-1B
base_model_name = "unsloth/Llama-3.2-1B"
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# チェックポイントモデル
checkpoint_dir = "output/20240315/checkpoint-4000"
ft_model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
ft_tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
base_model.to(device)
ft_model.to(device)

prompt = "どのようにSQLインジェクションを防ぐのか？"

base_inputs = base_tokenizer(prompt, return_tensors="pt")
ft_inputs = ft_tokenizer(prompt, return_tensors="pt")

base_inputs = {k: v.to(device) for k, v in base_inputs.items()}
ft_inputs = {k: v.to(device) for k, v in ft_inputs.items()}

generation_kwargs = {
    "max_length": 256,
    "num_return_sequences": 1,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9
}

base_outputs = base_model.generate(**base_inputs, **generation_kwargs)
ft_outputs = ft_model.generate(**ft_inputs, **generation_kwargs)

base_text = base_tokenizer.decode(base_outputs[0], skip_special_tokens=True)
ft_text = ft_tokenizer.decode(ft_outputs[0], skip_special_tokens=True)


print("=== unsloth/Llama-3.2-1B（ベースモデル）の推論結果 ===")
print(base_text)
print("\n=== チェックポイントモデル（ファインチューニング済み）の推論結果 ===")
print(ft_text)
