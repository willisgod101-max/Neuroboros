"""
Drosophila seed circuit for feeding behavior.
Implements a 100-neuron, 3-layer architecture:
- Layer 1: 20 Sensory neurons (Sugar, Water, Bitter, Salt)
- Layer 2: 50 Integration neurons (Inhibitory gating)
- Layer 3: 30 Motor neurons (MN9/MN6 equivalent)
"""
from brian2 import *
from hybrid_ouroboros.spiking.equations import LIF_EQUATIONS, DROSOPHILA_PARAMS
import numpy as np

def create_drosophila_feeding_circuit():
    """
    Create the Drosophila feeding circuit with 3 layers:
    - Sensory layer: 20 neurons (5 each for sugar, water, bitter, salt)
    - Integration layer: 50 neurons with inhibitory gating
    - Motor layer: 30 neurons (MN9/MN6 equivalent)
    
    Returns
    -------
    sensory_neurons : Brian2 NeuronGroup
        Sensory neuron group
    integration_neurons : Brian2 NeuronGroup
        Integration neuron group
    motor_neurons : Brian2 NeuronGroup
        Motor neuron group
    sensory_to_integration : Brian2 Synapses
        Excitatory synapses from sensory to integration
    integration_to_motor : Brian2 Synapses
        Excitatory synapses from integration to motor
    inhibition_synapses : Brian2 Synapses
        Inhibitory synapses within integration layer
    """
    # Neuron parameters
    tau_m = DROSOPHILA_PARAMS['tau_m']
    tau_g = DROSOPHILA_PARAMS['tau_g']
    tau_ref = DROSOPHILA_PARAMS['tau_ref']
    v0 = DROSOPHILA_PARAMS['v0']
    v_r = DROSOPHILA_PARAMS['v_r']
    v_th = DROSOPHILA_PARAMS['v_th']
    
    # Layer 1: Sensory neurons (20 total)
    # 5 each for sugar, water, bitter, salt
    n_sensory = 20
    sensory_neurons = NeuronGroup(n_sensory, LIF_EQUATIONS,
                                  threshold='v > v_th', reset='v = v_r',
                                  refractory=tau_ref, method='euler')
    sensory_neurons.v = v0
    sensory_neurons.g = 0 * volt
    
    # Layer 2: Integration neurons (50 total) with inhibitory gating
    n_integration = 50
    integration_neurons = NeuronGroup(n_integration, LIF_EQUATIONS,
                                      threshold='v > v_th', reset='v = v_r',
                                      refractory=tau_ref, method='euler')
    integration_neurons.v = v0
    integration_neurons.g = 0 * volt
    
    # Layer 3: Motor neurons (30 total) - MN9/MN6 equivalent
    n_motor = 30
    motor_neurons = NeuronGroup(n_motor, LIF_EQUATIONS,
                                threshold='v > v_th', reset='v = v_r',
                                refractory=tau_ref, method='euler')
    motor_neurons.v = v0
    motor_neurons.g = 0 * volt
    
    # Sensory to Integration connections (excitatory)
    # Each sensory neuron connects to 10 random integration neurons
    sensory_to_integration = Synapses(sensory_neurons, integration_neurons,
                                      model='w : 1', on_pre='g += w')
    sensory_to_integration.connect(p=0.4)  # 40% connectivity
    sensory_to_integration.w = '0.5 * volt'  # Excitatory weight
    
    # Integration to Motor connections (excitatory)
    # Each integration neuron connects to 5 random motor neurons
    integration_to_motor = Synapses(integration_neurons, motor_neurons,
                                    model='w : 1', on_pre='g += w')
    integration_to_motor.connect(p=0.3)  # 30% connectivity
    integration_to_motor.w = '0.4 * volt'  # Excitatory weight
    
    # Inhibitory connections within integration layer (lateral inhibition)
    # Each integration neuron inhibits 5 random neighbors
    inhibition_synapses = Synapses(integration_neurons, integration_neurons,
                                   model='w : 1', on_pre='g -= w')
    inhibition_synapses.connect(p=0.2)  # 20% connectivity
    inhibition_synapses.w = '0.3 * volt'  # Inhibitory weight
    # Remove self-connections
    inhibition_synapses.disable_connections()
    
    return sensory_neurons, integration_neurons, motor_neurons, \
           sensory_to_integration, integration_to_motor, inhibition_synapses

def simulate_feeding_circuit(duration=1000*ms, input_rate=50*Hz):
    """
    Simulate the Drosophila feeding circuit for a given duration.
    
    Parameters
    ----------
    duration : float, optional
        Simulation duration (default: 1000*ms)
    input_rate : float, optional
        Poisson input rate for sensory stimulation (default: 50*Hz)
        
    Returns
    -------
    monitors : dict
        Dictionary containing SpikeMonitor objects for each layer
    """
    # Create the circuit
    sensory_neurons, integration_neurons, motor_neurons, \
    sensory_to_integration, integration_to_motor, inhibition_synapses = create_drosophila_feeding_circuit()
    
    # Set up monitors
    sensory_monitor = SpikeMonitor(sensory_neurons)
    integration_monitor = SpikeMonitor(integration_neurons)
    motor_monitor = SpikeMonitor(motor_neurons)
    
    # Apply some sensory input (e.g., sugar stimulus)
    # Activate first 5 sensory neurons (sugar) with Poisson input
    sugar_input = PoissonGroup(5, rates=input_rate)
    sugar_to_sensory = Synapses(sugar_input, sensory_neurons[:5], model='w : 1', on_pre='g += w')
    sugar_to_sensory.connect(p=1.0)  # All-to-all
    sugar_to_sensory.w = '0.6 * volt'
    
    # Run simulation
    net = Network(sensory_neurons, integration_neurons, motor_neurons,
                  sensory_to_integration, integration_to_motor, inhibition_synapses,
                  sugar_input, sugar_to_sensory,
                  sensory_monitor, integration_monitor, motor_monitor)
    net.run(duration)
    
    return {
        'sensory': sensory_monitor,
        'integration': integration_monitor,
        'motor': motor_monitor
    }

if __name__ == '__main__':
    # Example usage
    print("Simulating Drosophila feeding circuit...")
    monitors = simulate_feeding_circuit(500*ms)
    print(f"Sensory spikes: {len(monitors['sensory'].t)}")
    print(f"Integration spikes: {len(monitors['integration'].t)}")
    print(f"Motor spikes: {len(monitors['motor'].t)}")"" 
