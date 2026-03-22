import sys
import time
import torch
import os
sys.path.append('core')
sys.path.append('.')
from hybrid_ouroboros.gpt.model import GPT as BitGPT, GPTConfig as ModelConfig
from data.loader import DataLoader, evaluate
from core.optimizer import build_optimizer
from hybrid_ouroboros.bitnet.bitlinear_a48 import replace_with_a48
from rich.console import Console

console = Console()

def train_phase_a():
    # 0. Create checkpoint directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # 1. Setup Configuration
    config = ModelConfig(
        vocab_size=8000,  # SentencePiece vocab size
        block_size=256,   # Context window
        n_layer=8, 
        n_head=8, 
        n_embd=256,
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    console.print(f"[blue]Using device: {device}[/blue]")
    
    # 2. Initialize 1-bit Model with a4.8 Upgrades
    model = BitGPT(config).to(device)
    replace_with_a48(model, sparsity_k=0.5) 
    console.print(f"[cyan]Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters[/cyan]")
    
    # 3. Build Hybrid Optimizer (Muon + AdamW)
    optimizer = build_optimizer(model, lr=3e-4, weight_decay=0.1)
    train_loader = DataLoader(split='train', batch_size=32, seq_len=256)
    
    # 4. Training Loop with Wall-Clock Budget
    TRAIN_BUDGET = 300  # 5-minute experiment limit (change to 30 for testing)
    start_time = time.time()
    step = 0
    
    console.print(f"[bold green]Starting VIPER Phase A Training...[/bold green]")
    model.train()
    
    while (time.time() - start_time) < TRAIN_BUDGET:
        x, y = train_loader.get_batch(device=device)
        
        # Forward Pass
        logits, loss = model(x, y)
        
        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 10 == 0:
            elapsed = time.time() - start_time
            console.print(f"Step {step} | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s")
        step += 1

    # 5. Final Evaluation
    val_bpb = evaluate(model, split='val')
    console.print(f"\n[bold cyan]Training Complete.[/bold cyan]")
    console.print(f"Final Metric: [yellow]val_bpb={val_bpb:.4f}[/yellow]")
    console.print(f"Total Steps: {step} | Time: {elapsed:.1f}s")
    
    # Save Checkpoint for NEUROBOROS Linguistic Pressure
    torch.save(model.state_dict(), 'checkpoints/gpt_latest.pt')
    console.print(f"[green]Checkpoint saved to checkpoints/gpt_latest.pt[/green]")

if __name__ == "__main__":
    train_phase_a()
    console.print(f"Final Metric: [yellow]val_bpb={val_bpb:.4f}[/yellow]")
    
    # Save Checkpoint for NEUROBOROS Linguistic Pressure
    torch.save(model.state_dict(), 'checkpoints/gpt_latest.pt')

if __name__ == "__main__":
    train_phase_a()
