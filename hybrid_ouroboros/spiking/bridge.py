import torch
from brian2 import *

class SNNBridge:
    def __init__(self, n_neurons=1000):
        print(f"🕷️ Initializing SNNBridge with {n_neurons} FlyWire neurons...")
        
        # Brian2 preferences for mobile/CPU
        prefs.codegen.target = 'numpy'
        
        # Drosophila-grounded parameters from FlyWire
        v_r = -65 * mV
        v_th_base = -50 * mV
        v_reset = -65 * mV
        tau_V = 10 * ms
        
        # Leaky Integrate-and-Fire equations with linguistic_pressure modulation
        eqs = '''
        dv/dt = (v_r - v) / tau_V : volt
        linguistic_pressure : volt  # GPT loss → firing threshold modulation
        rate : Hz
        '''
        
        self.group = NeuronGroup(n_neurons, eqs, threshold=f'v > v_th_base + linguistic_pressure', 
                                reset='v = v_reset', method='euler')
        self.group.v = v_r + rand(n_neurons) * (v_th_base - v_r)
        self.group.linguistic_pressure = 0 * mV
        self.group.rate = 5 * Hz  # Baseline
        
        # Simple recurrent synapses for network dynamics
        self.synapses = Synapses(self.group, self.group, 
                                'w : 1', on_pre='v_post += w * 0.5 * mV')
        self.synapses.connect(p=0.1)
        self.synapses.w = '0.1 * rand()'
        
        self.net = Network(self.group, self.synapses)
        self.spike_monitor = SpikeMonitor(self.group)
        
        print("✅ SNNBridge ready for GPT fusion.")
    
    def step(self, pressure):
        # Update threshold from GPT Linguistic Pressure (0.0-1.0 → voltage)
        self.group.linguistic_pressure = pressure * 10 * mV  # Scale to modulate firing
        
        # Run one simulation timestep
        self.net.run(1 * ms)
        
        # Compute behavioral fitness stats
        total_spikes = len(self.spike_monitor.t)
        mean_rate = total_spikes / self.group.N * 1000  # Hz
        target_rate = 10 * Hz
        behavior_acc = min(mean_rate / target_rate, 1.0)  # 0-1 fitness
        
        # Clear for next step
        self.spike_monitor.reset()
        
        return {
            'behavior_acc': float(behavior_acc),
            'mean_rate': float(mean_rate),
            'total_spikes': int(total_spikes),
            'pressure_applied': float(pressure)
        }

