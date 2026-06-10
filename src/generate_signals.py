import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Ensure data/ directory exists
    os.makedirs("data", exist_ok=True)
    
    # Generate time/X-axis
    t = np.linspace(0, 10, 1000)
    
    # Generate a clean signal from several sine waves
    signal = 2 * np.sin(2 * np.pi * 1.5 * t) + np.sin(2 * np.pi * 0.5 * t) + 0.5 * np.cos(2 * np.pi * 3 * t)
    
    # Inject a secret malfunction: an ugly high-frequency clipping artifact caused by amplitude saturation
    # Randomly pick a start index, ensuring there's room for the anomaly
    malfunction_start_idx = np.random.randint(100, 900)
    malfunction_end_idx = malfunction_start_idx + np.random.randint(20, 50)
    
    # Generate high frequency noise and clip it to simulate saturation
    noise = np.random.uniform(-4, 4, malfunction_end_idx - malfunction_start_idx)
    noise = np.clip(noise, -2.5, 2.5)
    
    # Add the noise to the signal
    signal[malfunction_start_idx:malfunction_end_idx] += noise
    
    # Plot the signal
    plt.figure(figsize=(12, 6))
    plt.plot(t, signal, color='#1f77b4', linewidth=1.5, label='Telemetry Signal')
    plt.title("System Telemetry Waveform Analysis")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the plot
    save_path = os.path.join("data", "audit_target.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot successfully saved to {save_path}")

if __name__ == "__main__":
    main()
