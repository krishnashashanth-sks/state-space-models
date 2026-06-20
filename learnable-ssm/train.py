def train(num_epochs,observed_data,ssm_model,optimizer,criterion,obs_dim):
    print("Starting Training...")
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass through the SSM (Kalman Filter) to get innovations and their covariances
        estimated_states, estimated_covariances, innovations, innovation_covariances = ssm_model(observed_data)

        # Calculate Negative Log-Likelihood (NLL) as the loss
        loss = criterion(innovations, innovation_covariances, obs_dim)

        loss.backward() # Compute gradients
        optimizer.step() # Update parameters

        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    print("Training Complete!")