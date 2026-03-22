import sys
sys.path.append('core')
sys.path.append('.')
from hybrid_ouroboros.gpt.model import GPT as BitGPT, GPTConfig as ModelConfig
from data.loader import DataLoader, evaluate
from core.optimizer import build_optimizer
from bitnet.bitlinear_a48 import replace_with_a48
from rich.console import Console

console = Console()

def train_phase_a():
    # 1. Setup Configuration
    config = ModelConfig(n_layer=8, n_head=8, n_embd=256) # As per spec
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 2. Initialize 1-bit Model with a4.8 Upgrades
    model = BitGPT(config).to(device)
    replace_with_a48(model, sparsity_k=0.5) 
    
    # 3. Build Hybrid Optimizer (Muon + AdamW)
    optimizer = build_optimizer(model, lr=3e-4, weight_decay=0.1)
    train_loader = DataLoader(split='train')
    
    # 4. Training Loop with Wall-Clock Budget
    TRAIN_BUDGET = 300 # 5-minute experiment limit
    start_time = time.time()
    step = 0
    
    console.print(f"[bold green]Starting VIPER Phase A Training...[/bold green]")
    model.train()
    
    while (time.time() - start_time) < TRAIN_BUDGET:
        x, y = train_loader.get_batch(device)
        
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
    
    # Save Checkpoint for NEUROBOROS Linguistic Pressure
    torch.save(model.state_dict(), 'checkpoints/gpt_latest.pt')

if __name__ == "__main__":
    train_phase_a()
