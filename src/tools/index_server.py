from aiohttp import web
import torch
import logging
import os
from qt_state_processor import QuantumStateProcessor
from aiohttp.web_exceptions import HTTPInternalServerError, HTTPNotFound
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# File paths for saving quantum data
quantum_data_dir = "quantum_data"
os.makedirs(quantum_data_dir, exist_ok=True)

class QuantumOptimizedServer:
    def __init__(self):
        self.qsp = QuantumStateProcessor(n_qubits=2)
        self.recent_state = None

    async def handle_request(self, request):
        try:
            request_data = await request.json()
            alpha = request_data.get("alpha", 1 / np.sqrt(2))
            beta = request_data.get("beta", 1 / np.sqrt(2))

            # Create a superposition state and measure it
            quantum_state = self.qsp.create_superposition(alpha, beta)
            measurements = self.qsp.measure_state(quantum_state)

            # Calculate entanglement entropy for demonstration
            entanglement_entropy = self.qsp.get_entanglement_entropy(quantum_state, partition=1)
            self.recent_state = {
                "state_real": quantum_state.real.tolist(),
                "state_imag": quantum_state.imag.tolist(),
                "measurements": measurements.tolist(),
                "entanglement_entropy": entanglement_entropy,
            }

            # Save data here
            np.save(os.path.join(quantum_data_dir, "measurements.npy"), measurements.numpy())
            with open(os.path.join(quantum_data_dir, "entanglement_entropy.json"), "w") as f:
                f.write(f"{entanglement_entropy}")

            # Log quantum data for real-time extraction
            logger.debug(f"Quantum state data: {self.recent_state}")
            return self.recent_state
        except Exception as e:
            logger.error(f"Failed to process request: {e}")
            raise HTTPInternalServerError()

# Server Routes
async def handle_index(request):
    file_path = os.path.join(os.path.dirname(__file__), 'index.html')
    if not os.path.exists(file_path):
        raise HTTPNotFound(reason="index.html not found.")
    with open(file_path, 'r') as f:
        return web.Response(text=f.read(), content_type="text/html")

async def handle_complex(request):
    quantum_server = QuantumOptimizedServer()
    try:
        q_state = await quantum_server.handle_request(request)
        return web.json_response(q_state)
    except Exception as e:
        logger.error(f"Failed to process /complex request: {e}")
        return web.Response(text="Internal Server Error", status=500)

# New endpoint to provide recent quantum data
async def handle_quantum_state(request):
    quantum_server = QuantumOptimizedServer()
    if quantum_server.recent_state:
        return web.json_response(quantum_server.recent_state)
    else:
        return web.json_response({"error": "No quantum data available yet."})

# Set up and run the aiohttp web server
app = web.Application()
app.add_routes([
    web.get('/', handle_index),
    web.post('/complex', handle_complex),
    web.get('/quantum_state', handle_quantum_state)  # New route for real-time quantum data
])

web.run_app(app, port=8080, handle_signals=True)
