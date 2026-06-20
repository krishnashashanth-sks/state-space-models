from tqdm.auto import tqdm
import torch
def train(num_epochs,model,train_dataloader,optimizer,criterion,device):
    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss = 0

        for batch in tqdm(train_dataloader):
            # Move batch to device
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Remove this line, as nn.Embedding expects Long type
            # inputs = inputs.float()

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()

            # Gradient clipping is recommended for S4/SSM models
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")