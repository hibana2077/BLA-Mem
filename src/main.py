import argparse
import sys
import torch
import torch.nn as nn
import time
from src.data import ParityDataset, AddingDataset
from src.models.gssm import GSSM
from src.models.baselines import SimpleMamba, TransformerBaseline

def get_model(args, device):
    if args.task == 'parity':
        d_in = 1
    elif args.task == 'adding':
        d_in = 2
    else:
        raise ValueError(f"Unknown task: {args.task}")

    # Ensure max_len covers both train and val seq_len for Transformer PE
    val_len = args.val_seq_len if args.val_seq_len else args.seq_len
    max_seq_len = max(args.seq_len, val_len)

    if args.model == 'gssm':
        return GSSM(d_in=d_in, d_model=args.d_model, k_group=args.d_model//2).to(device)
    elif args.model == 'mamba':
        return SimpleMamba(d_in=d_in, d_model=args.d_model).to(device)
    elif args.model == 'transformer':
        # Allocate enough PE for the longest sequence we might see
        return TransformerBaseline(d_in=d_in, d_model=args.d_model, max_len=max_seq_len * 2).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    val_seq_len = args.val_seq_len if args.val_seq_len is not None else args.seq_len

    # Setup data
    start_seq_len = args.min_seq_len if args.curriculum else args.seq_len

    if args.task == 'parity':
        train_ds = ParityDataset(seq_len=start_seq_len, batch_size=args.batch_size, device=device)
        val_ds = ParityDataset(seq_len=val_seq_len, batch_size=args.batch_size, device=device)
        criterion = nn.CrossEntropyLoss()
    elif args.task == 'adding':
        train_ds = AddingDataset(seq_len=start_seq_len, batch_size=args.batch_size, device=device)
        val_ds = AddingDataset(seq_len=val_seq_len, batch_size=args.batch_size, device=device)
        criterion = nn.MSELoss()
    
    train_iter = iter(train_ds)
    val_iter = iter(val_ds)
    
    # Setup model
    model = get_model(args, device)
    print(f"Model: {args.model}, Task: {args.task}, SeqLen: {args.seq_len}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training Loop
    start_time = time.time()
    
    for step in range(1, args.steps + 1):
        if args.curriculum:
            # Linearly increase seq_len from min_seq_len to seq_len
            progress = (step - 1) / args.steps
            current_len = int(args.min_seq_len + (args.seq_len - args.min_seq_len) * progress)
            train_ds.seq_len = current_len

        model.train()
        x, y = next(train_iter)
        
        optimizer.zero_grad()
        parity_logits, add_pred = model(x)
        
        if args.task == 'parity':
            loss = criterion(parity_logits, y)
        else: # adding
            loss = criterion(add_pred.squeeze(), y)
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Logging
        if step % args.val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_x, val_y = next(val_iter)
                p_logits, a_pred = model(val_x)
                
                if args.task == 'parity':
                    val_loss = criterion(p_logits, val_y)
                    preds = torch.argmax(p_logits, dim=1)
                    acc = (preds == val_y).float().mean().item()
                    metric_str = f"Acc: {acc:.4f}"
                else:
                    val_loss = criterion(a_pred.squeeze(), val_y)
                    metric_str = f"MSE: {val_loss.item():.6f}"
            
            elapsed = time.time() - start_time
            cur_len_str = f"| Len: {train_ds.seq_len} " if args.curriculum else ""
            print(f"Step {step}/{args.steps} {cur_len_str}| Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f} | {metric_str} | Time: {elapsed:.1f}s")
    
    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=['gssm', 'mamba', 'transformer'])
    parser.add_argument("--task", type=str, required=True, choices=['parity', 'adding'])
    parser.add_argument("--seq_len", type=int, default=100, help="Target sequence length (end length if curriculum)")
    parser.add_argument("--val_seq_len", type=int, default=None, help="Sequence length for validation. Defaults to seq_len if not set.")
    parser.add_argument("--curriculum", action="store_true", help="Use sequence length curriculum")
    parser.add_argument("--min_seq_len", type=int, default=40, help="Start sequence length for curriculum")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--val_interval", type=int, default=200, help="Interval for validation and logging")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    
    args = parser.parse_args()
    train(args)
