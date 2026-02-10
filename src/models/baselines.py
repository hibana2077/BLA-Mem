import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMamba(nn.Module):
    """
    A pure PyTorch implementation of the selective state space part of Mamba (S6).
    Without the hardware-aware scan, this is just a gated RNN.
    Used as a baseline.
    """
    def __init__(self, d_in, d_model=64):
        super().__init__()
        self.d_model = d_model
        
        # Input projection/Expansion (simplified)
        self.x_proj = nn.Linear(d_in, d_model)
        
        # SSM parameters
        self.to_lambda = nn.Linear(d_in, d_model)
        self.to_delta  = nn.Linear(d_in, d_model)
        self.to_B      = nn.Linear(d_in, d_model) # In strict mamba B is input dependent
        self.to_C      = nn.Linear(d_in, d_model) # C is input dependent

        # Allow C to project back to readout or just use state directly for task heads
        # To make it comparable to GS-SSM structure for these specific tasks:
        self.parity_head = nn.Linear(d_model, 2)
        self.add_head = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x: (B, T, d_in)
        """
        B, T, _ = x.shape
        device = x.device
        
        s = torch.zeros(B, self.d_model, device=device)
        
        # If we strictly followed Mamba, we would have a Conv1d here potentially.
        # But we stick to the core recurrence: s_t = A_t * s_{t-1} + B_t * x_t
        # Where A_t is discretization of A (diagonal) with delta_t.
        
        for t in range(T):
            xt = x[:, t, :]
            
            # Project input
            x_proj = self.x_proj(xt)
            
            # Calculate input-dependent dynamics
            lam = F.softplus(self.to_lambda(xt)) # "A" parameter magnitude (d_model)
            delt = F.softplus(self.to_delta(xt)) # Step size
            
            # Discretize
            # A_bar = exp(-delta * lambda)
            # B_bar = delta * B (here we simplify B to be input generated directly or static. 
            # Let's input-depend B like the 'inp' in GSSM for fairness)
            
            input_signal = self.to_B(xt) # This acts as the "x * B" part
            
            alpha = torch.exp(-delt * lam)
            
            # Update state
            s = alpha * s + delt * input_signal
            
        # Readout from final state
        # In full Mamba, y_t = C_t * s_t. Here we just take s_T.
        
        parity_logits = self.parity_head(s)
        add_pred = self.add_head(s)
        
        return parity_logits, add_pred


class TransformerBaseline(nn.Module):
    """
    Standard Transformer Encoder.
    """
    def __init__(self, d_in, d_model=64, n_heads=4, n_layers=2, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(d_in, d_model)
        
        # Positional Encoding
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Heads
        self.parity_head = nn.Linear(d_model, 2)
        self.add_head = nn.Linear(d_model, 1)

    def forward(self, x):
        B, T, _ = x.shape
        x_emb = self.embedding(x)
        
        # Add PE
        if T > self.pe.shape[1]:
           x_emb = x_emb + self.pe[:, :T, :] # Simplistic handling, might fail if huge
        else:
           x_emb = x_emb + self.pe[:, :T, :]

        # Transformer
        # (B, T, D)
        out = self.transformer(x_emb)
        
        # Pool or take last
        # For sequence tasks, taking the last token is common if causal, 
        # but standard TransformerEncoder is bidirectional unless masked. 
        # We'll use the last token representation as the summary.
        last_hidden = out[:, -1, :]
        
        parity_logits = self.parity_head(last_hidden)
        add_pred = self.add_head(last_hidden)
        
        return parity_logits, add_pred
