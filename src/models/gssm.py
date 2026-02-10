import torch
import torch.nn as nn
import torch.nn.functional as F

class GSSM(nn.Module):
    def __init__(self, d_in, d_model=64, k_group=8, n_layers=1):
        super().__init__()
        # For simplicity in this toy implementation, we combine layers into a single iterative block
        # or just use one layer as shown in the idea snippet. 
        # The prompt implies a direct implementation of the snippet.
        
        self.d_model = d_model
        self.k_group = k_group
        
        # --- Group Memory Branch (U(1)) ---
        # Projects input to 'k_group' angles
        self.theta_proj = nn.Linear(d_in, k_group)

        # --- Selective Euclidean Branch (Simplified Mamba) ---
        # Projects input to state dynamics parameters
        self.to_lambda = nn.Linear(d_in, d_model)
        self.to_delta  = nn.Linear(d_in, d_model)
        self.to_inp    = nn.Linear(d_in, d_model)

        # --- Readouts ---
        # Parity often uses the geometric state
        self.parity_head = nn.Linear(2 * k_group, 2) 
        # Adding uses the euclidean state
        self.add_head = nn.Linear(d_model, 1)
        
        # General projection for mixing if needed, but we follow the snippet's specific heads for specific tasks
        # If the task is "parity", we largely rely on parity_head. 
        # Ideally, a general model would mix them. 
        # Let's make a combined head for general usage if needed, 
        # but for this specific "Methodology" section, it separates them or concatenates.
        self.output_proj = nn.Linear(2 * k_group + d_model, d_in) # Dummy for sequence-to-sequence if needed

    def forward(self, x):
        """
        x: (B, T, d_in)
        Returns: 
           parity_logits: (B, 2)
           add_pred: (B, 1)
        """
        B, T, _ = x.shape
        device = x.device

        # Identity element for U(1) is 1 + 0i
        g_re = torch.ones(B, self.k_group, device=device)
        g_im = torch.zeros(B, self.k_group, device=device)

        # Euclidean state initialized to 0
        s = torch.zeros(B, self.d_model, device=device)

        # To support stacking layers, we would need to emit a sequence.
        # But this is a simple RNN loop implementation for the specific tasks.
        
        # We will collect states if we wanted seq-to-seq, but for parity/adding we only need the last state.
        
        for t in range(T):
            xt = x[:, t, :]

            # --- (A) Group Update: g <- g * exp(i * pi * theta(x)) ---
            # The idea doc says: theta_t = pi * u_t. 
            # We assume theta_proj outputs 'u_t' or the scaled value. 
            # Let's align with doc: "theta_t = pi * select". 
            # We'll treat the linear output as the selection factor.
            th = torch.pi * self.theta_proj(xt) 
            c, s_th = torch.cos(th), torch.sin(th)

            # Complex multiply: (re + i*im) * (c + i*s) = (re*c - im*s) + i(re*s + im*c)
            new_re = g_re * c - g_im * s_th
            new_im = g_re * s_th + g_im * c
            g_re, g_im = new_re, new_im

            # --- (B) Selective Euclidean Update ---
            # Mamba style: s_t = alpha_t * s_{t-1} + beta_t * input
            # alpha = exp(-delta * softplus(lambda))
            # beta = delta
            
            lam = F.softplus(self.to_lambda(xt))
            delt = F.softplus(self.to_delta(xt))
            
            # alpha must be in (0, 1]. exp(-positive) is in (0, 1].
            alpha = torch.exp(-delt * lam) 
            
            inp_val = self.to_inp(xt)
            
            # Update s
            s = alpha * s + delt * inp_val

        # --- (C) Readout (Last Step) ---
        g_feat = torch.cat([g_re, g_im], dim=-1) # (B, 2*k)
        
        parity_logits = self.parity_head(g_feat)
        add_pred = self.add_head(s)

        return parity_logits, add_pred
