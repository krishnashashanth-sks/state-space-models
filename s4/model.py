import torch.nn as nn
from layers import S4Layer

class S4Model(nn.Module):
    def __init__(self, vocab_size, d_model, state_dim, num_classes, num_layers, L_max, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.L_max = L_max

        # Embedding layer to convert token IDs to d_model features
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Stack multiple S4 Layers
        self.s4_layers = nn.ModuleList([
            S4Layer(d_model=d_model, state_dim=state_dim, L=L_max, dt_min=dt_min, dt_max=dt_max)
            for _ in range(num_layers)
        ])

        # Final classification head
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len) - token IDs are expected now

        # 1. Embed input tokens
        x = self.embedding(x) # Output shape: (batch_size, seq_len, d_model)

        # 2. Permute to (batch_size, d_model, seq_len) for S4Layer
        x = x.permute(0, 2, 1) # Output shape: (batch_size, d_model, seq_len)

        # 3. Pass through S4 Layers
        for layer in self.s4_layers:
            x = layer(x) # Output shape: (batch_size, d_model, seq_len)

        # 4. Global Average Pooling to get a fixed-size representation
        # Average over the sequence length dimension
        x = x.mean(dim=-1) # Output shape: (batch_size, d_model)

        # 5. Classification head
        logits = self.classifier(x) # Output shape: (batch_size, num_classes)

        return logits