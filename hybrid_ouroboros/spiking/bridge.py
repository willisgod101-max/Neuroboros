import torch
from brian2 import *
from .equations import LIF_EQUATIONS, DROSOPHILA_PARAMS
from hybrid_ouroboros.gpt.init_system import NeuroGenesis  # For FlyWire substrate

class SNNBridge:
    def __init__(self, n_neurons=1000):
        print(f"🕷️ Initializing SNNBridge with {n_neurons} FlyWire neurons...")
        
        # Brian2 preferences for mobile/CPU
        prefs.codegen.target = 'numpy'
        
        self.params = DROSOPHILA_PARAMS
        # v_reset aligns with v_r for reset
        
        # Full LIF with conductance from FlyWire/Drosophila
        eqs = LIF_EQUATIONS + '''
        linguistic_pressure : volt  # GPT loss → firing threshold modulation
        rate : Hz
        '''
        
        self.group = NeuronGroup(n_neurons, eqs, namespace=self.params,
                                threshold='v > v_th + linguistic_pressure',
                                reset='v = v_r',
                                refractory=self.params['tau_ref'],
                                method='euler')
        # Ground to FlyWire substrate (randomize to break synchrony)
        NeuroGenesis.initialize_flywire_substrate(self, n_neurons)
        self.group.linguistic_pressure = 0 * volt
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
        self.group.linguistic_pressure = pressure * self.params['v_th']  # Biological scale from FlyWire
        
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

