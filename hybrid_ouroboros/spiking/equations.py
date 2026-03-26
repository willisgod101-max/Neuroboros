from brian2 import *

# LIF Equations
LIF_EQUATIONS = '''
dv/dt = (v0 - v + g) / tau_m : volt (unless refractory)
dg/dt = -g / tau_g            : volt (unless refractory)
'''

# Drosophila parameters from FlyWire connectome
DROSOPHILA_PARAMS = {
    'tau_m'  : 20  * ms,    # Membrane time constant
    'tau_g'  : 5   * ms,    # Conductance time constant
    'tau_ref': 2.2 * ms,    # Refractory period
    'v0'     : 0   * mV,    # Resting potential
    'v_r'    : 0   * mV,    # Reset potential
    'v_th'   : 7   * mV,    # Spike threshold
}