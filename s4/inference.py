import torch

def predict_sentiment(text, model, tokenizer, device):
    model.eval()
    # 1. Preprocess & Tokenize
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 2. Forward Pass (No Gradients)
    with torch.no_grad():
        logits = model(inputs['input_ids'])
        prediction = torch.argmax(logits, dim=-1)

    return "Positive" if prediction.item() == 1 else "Negative"
