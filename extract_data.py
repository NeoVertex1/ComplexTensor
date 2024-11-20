import numpy as np
import os

# Directory containing .npy files
output_dir = "complex_embeddings_outputs"
log_file = "extracted_data.log"

def load_and_log_npy_files(directory, log_filename):
    with open(log_filename, "w") as log:
        log.write("Extracted Data from .npy Files:\n\n")
        for filename in os.listdir(directory):
            if filename.endswith(".npy"):
                filepath = os.path.join(directory, filename)
                data = np.load(filepath)
                
                # Write summary information
                log.write(f"File: {filename}\n")
                log.write(f"Shape: {data.shape}\n")
                
                # Write smaller chunks of data if it's large
                if data.size > 100:  # Adjust threshold as needed
                    log.write(f"Sample (first 100 elements): {data.flat[:100].tolist()}\n")
                else:
                    log.write(f"Data: {data.tolist()}\n")
                
                log.write("\n")
    print(f"Data extracted and saved to {log_filename}")

# Extract data from .npy files and log
load_and_log_npy_files(output_dir, log_file)
