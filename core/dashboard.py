class NeuroborosDashboard:
    def __init__(self):
        print("📊 NEUROBOROS Dashboard Online")
    
    def update(self, astro, pressure, step, epoch=0):
        print(f"Step {step} | Evo: {astro.evolutionary_surplus:.2f} | Lin: {astro.linguistic_surplus:.2f} | Sys: {astro.systemic_surplus:.2f} | Pressure: {pressure:.2f}")

