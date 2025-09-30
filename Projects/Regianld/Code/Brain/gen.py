
# filename: gen.py

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def main():
    # —— Hard-coded settings —— 
    model_dir            = "Regie"
    prompt               = "Create a role that is green"
    max_length           = 200
    temperature          = 0.8
    top_p                = 0.9
    num_return_sequences = 1

    # 1) Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2) Load tokenizer & model
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model     = GPT2LMHeadModel.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # 3) Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # 4) Generate
    output_ids = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.pad_token_id,
    )

    # 5) Decode and manually truncate at '<EOS>' string
    prompt_len = input_ids.size(1)
    for i, sequence in enumerate(output_ids, start=1):
        tokens = sequence.tolist()
        new_tokens = tokens[prompt_len:]  # strip prompt
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Truncate at custom EOS string
        if "<EOS>" in text:
            text = text.split("<EOS>", 1)[0]

        print(f"\n=== GENERATED #{i} ===")
        print(f"Generated {len(new_tokens)} tokens (before truncation)")
        print(f"{text.strip()}\n")

if __name__ == "__main__":
    main()
