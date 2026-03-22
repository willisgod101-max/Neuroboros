import numpy as np
from multiprocessing import shared_memory

class SystemBridge:
    def __init__(self, name, size):
        try:
            self.shm = shared_memory.SharedMemory(create=True, size=size, name=name)
        except FileExistsError:
            raise RuntimeError(
                f"Shared memory block '{name}' already exists. "
                "Use a different name or clean up the existing block."
            )
    
    def send_spike_data(self, data: np.ndarray):
        # Ensure the data fits
        if data.nbytes > self.shm.size:
            raise ValueError("Data too large for shared memory block")
        self.shm.buf[:data.nbytes] = data.tobytes()
    
    def close(self):
        self.shm.close()
        self.shm.unlink()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()