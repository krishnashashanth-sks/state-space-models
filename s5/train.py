import torch

def train(num_epochs,synthetic_input_train,synthetic_target,s5_model,optimizer,criterion):
    for epoch in range(num_epochs):
        s5_model.train() # Set model to training mode
        optimizer.zero_grad() # Zero the gradients before each epoch

        # Forward pass
        output = s5_model(synthetic_input_train)

        # Calculate loss
        loss = criterion(output, synthetic_target)

        # Backward pass and optimization
        loss.backward()
        # Add gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(s5_model.parameters(), max_norm=1.0)
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
