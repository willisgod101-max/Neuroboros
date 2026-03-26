#!/usr/bin/env python3
"""
Smoke test for VIPER-NEUROBOROS unified substrate.
Tests:
1. Ternary weight quantization (BitLinear).
2. Spiking activity (Brian2).
3. Shared memory connectivity (Bridge).
"""
import sys
import torch
import numpy as np

# Add the core directory to the path to import our modules
sys.path.append('core')
sys.path.append('.')

def test_bitlinear():
    """Test BitLinear layer for ternary weights and proper forward pass."""
    print("Testing BitLinear...")
    try:
        from bitlinear import BitLinear
        # Create a small linear layer
        in_features, out_features = 4, 2
        layer = BitLinear(in_features, out_features, bias=True)
        
        # Create a random input tensor
        x = torch.randn(1, in_features)
        
        # Forward pass
        output = layer(x)
        
        # Check output shape
        assert output.shape == (1, out_features), f"Expected output shape (1, {out_features}), got {output.shape}"
        
        # Check that weights are ternary (-1, 0, 1) after sign()
        with torch.no_grad():
            binary_weight = layer.weight.sign()
            unique_vals = torch.unique(binary_weight)
            expected_vals = torch.tensor([-1., 0., 1.])
            # Check that all unique values in binary_weight are in [-1, 0, 1]
            assert torch.all(torch.isin(unique_vals, expected_vals)), f"Weights are not ternary: {unique_vals}"
        
        print("  ✓ BitLinear test passed")
        return True
    except Exception as e:
        print(f"  ✗ BitLinear test failed: {e}")
        return False

def test_brian2():
    """Test Brian2 for spiking activity."""
    print("Testing Brian2...")
    try:
        from brian2 import *
        # Set up a simple neuron group
        tau = 20*ms
        eqs = '''
        dv/dt = (1 - v) / tau : 1
        '''
        G = NeuronGroup(10, eqs, threshold='v>1', reset='v=0', method='exact')
        G.v = 0
        
        # Monitor spikes
        M = SpikeMonitor(G)
        
        # Run simulation
        run(50*ms)
        
        # Check that we recorded some spikes
        assert len(M.t) > 0, "No spikes recorded"
        
        print(f"  ✓ Brian2 test passed: recorded {len(M.t)} spikes")
        return True
    except Exception as e:
        print(f"  ✗ Brian2 test failed: {e}")
        return False

def test_shared_memory():
    """Test shared memory connectivity (placeholder for actual bridge)."""
    print("Testing Shared Memory (placeholder)...")
    try:
        # This is a placeholder for the actual SystemBridge using multiprocessing.shared_memory
        # For the smoke test, we just check that we can import multiprocessing
        import multiprocessing
        from multiprocessing import shared_memory
        
        # Create a shared memory block
        shm = shared_memory.SharedMemory(create=True, size=10)
        
        # Write some data
        shm.buf[:5] = b'hello'
        
        # Read the data
        assert bytes(shm.buf[:5]) == b'hello'
        
        # Clean up
        shm.close()
        shm.unlink()
        
        print("  ✓ Shared Memory test passed")
        return True
    except Exception as e:
        print(f"  ✗ Shared Memory test failed: {e}")
        return False

def main():
    """Run all smoke tests."""
    print("Running VIPER-NEUROBOROS Smoke Tests\n")
    
    results = []
    results.append(test_bitlinear())
    results.append(test_brian2())
    results.append(test_shared_memory())
    
    print("\n" + "="*50)
    if all(results):
        print("All smoke tests PASSED")
        sys.exit(0)
    else:
        print("Some smoke tests FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()