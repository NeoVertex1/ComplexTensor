import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from complextensor.complex_tensor import ComplexTensor
from src.complextensor.apply_gate import apply_gate
import os


def create_bell_state():
    """
    Simulate the creation of a Bell state using tensor products and gate applications.

    Returns:
        ComplexTensor: Final entangled Bell state.
    """
    # Define |0> and |1> states
    zero = ComplexTensor(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 0.0]))
    one = ComplexTensor(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 0.0]))

    # Step 1: Create initial state |00> using tensor product
    qubit_1 = zero
    qubit_2 = zero
    initial_state_real = torch.outer(qubit_1.real, qubit_2.real)  # Tensor product for real
    initial_state_imag = torch.outer(qubit_1.imag, qubit_2.imag)  # Tensor product for imaginary
    initial_state = ComplexTensor(initial_state_real, initial_state_imag)

    print("[INFO] Initial State |00>:")
    print(f"Real Part:\n{initial_state.real}")
    print(f"Imaginary Part:\n{initial_state.imag}")

    # Step 2: Apply Hadamard gate to the first qubit
    hadamard_gate = torch.tensor([[1.0, 1.0], [1.0, -1.0]]) / torch.sqrt(torch.tensor(2.0))
    hadamard_gate_2q = torch.kron(hadamard_gate, torch.eye(2))  # Expand Hadamard gate to 2-qubit space

    superposed_state = apply_gate(initial_state, hadamard_gate_2q)

    print("[INFO] After Hadamard Gate on Qubit 1:")
    print(f"Real Part:\n{superposed_state.real}")
    print(f"Imaginary Part:\n{superposed_state.imag}")

    # Step 3: Apply CNOT gate
    cnot_gate = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=torch.float32)

    entangled_state = apply_gate(superposed_state, cnot_gate)

    print("[INFO] Entangled Bell State:")
    print(f"Real Part:\n{entangled_state.real}")
    print(f"Imaginary Part:\n{entangled_state.imag}")

    return entangled_state


def visualize_entangled_state(state, output_dir="output", filename="bell_state.png"):
    """
    Visualize the real and imaginary parts of an entangled state as a 3D bar plot and save it to a file.

    Args:
        state (ComplexTensor): The quantum state to visualize.
        output_dir (str): Directory where the plot will be saved.
        filename (str): Name of the file to save the plot.
    """
    # Flatten the probabilities to a 1D list
    probabilities = (state.real**2 + state.imag**2).flatten()
    labels = ["|00⟩", "|01⟩", "|10⟩", "|11⟩"]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    x = list(range(len(labels)))
    y = [0] * len(labels)  # Single row
    z = [0] * len(labels)
    dx = dy = [0.5] * len(labels)
    dz = probabilities.tolist()

    # Add real and imaginary components as color gradients
    colors = ["blue" if prob > 0.5 else "red" for prob in dz]

    ax.bar3d(x, y, z, dx, dy, dz, color=colors, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("State")
    ax.set_ylabel("Amplitude")
    ax.set_zlabel("Probability")
    ax.set_title("Quantum Bell State Probability Distribution")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Saving files to: {os.path.abspath(output_dir)}")
    filepath = os.path.join(output_dir, filename)

    # Save the plot to a file
    try:
        plt.savefig(filepath)
        print(f"[INFO] 3D visualization saved to {filepath}")
    except Exception as e:
        print(f"[ERROR] Could not save the plot to {filepath}: {e}")

    plt.show()
    plt.close()  # Close the figure to free up memory


def save_state_data(state, output_dir="output", filename="bell_state_data.txt"):
    """
    Save the real and imaginary parts of the state as a human-readable text file.

    Args:
        state (ComplexTensor): The quantum state to save.
        output_dir (str): Directory where the data will be saved.
        filename (str): Name of the file to save the data.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    # Write data to file
    try:
        with open(filepath, "w") as f:
            f.write("[INFO] Bell State Data\n")
            f.write("Real Part:\n")
            f.write(f"{state.real.numpy()}\n")
            f.write("Imaginary Part:\n")
            f.write(f"{state.imag.numpy()}\n")
        print(f"[INFO] State data saved to {filepath}")
    except Exception as e:
        print(f"[ERROR] Could not save the state data to {filepath}: {e}")


if __name__ == "__main__":
    # Create the Bell state
    bell_state = create_bell_state()

    # Visualize the resulting state
    visualize_entangled_state(bell_state)

    # Save the state data
    save_state_data(bell_state)
