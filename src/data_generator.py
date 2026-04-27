import os
import numpy as np
import matplotlib.pyplot as plt

def generate_square_wave_fourier(T=66, num_periods=100, num_harmonics=9, num_points_per_period=1000):
    """
    Generates the continuous Fourier series of a square wave.
    
    Args:
        T (float): Period of the square wave.
        num_periods (int): Number of periods to generate.
        num_harmonics (int): Number of odd harmonics to use.
        num_points_per_period (int): Number of time points per period.
        
    Returns:
        t (numpy.ndarray): Time vector.
        y (numpy.ndarray): Generated square wave amplitude.
    """
    # Calculate fundamental angular frequency
    omega_0 = 2 * np.pi / T
    
    # Generate time vector for the specified number of periods
    total_time = num_periods * T
    total_points = num_periods * num_points_per_period
    t = np.linspace(0, total_time, total_points, endpoint=False)
    
    # Initialize the amplitude array
    y = np.zeros_like(t)
    
    # Calculate the Fourier series using the first 'num_harmonics' odd harmonics
    # For a square wave alternating between 1 and -1, the Fourier series is:
    # y(t) = (4/pi) * (sin(omega_0*t) + (1/3)*sin(3*omega_0*t) + (1/5)*sin(5*omega_0*t) + ...)
    for i in range(num_harmonics):
        n = 2 * i + 1  # Generate odd harmonic numbers: 1, 3, 5, 7, 9, ...
        y += (4 / (n * np.pi)) * np.sin(n * omega_0 * t)
        
    return t, y

def generate_rc_filtered_square_wave(T=66, num_periods=100, num_harmonics=9, num_points_per_period=1000, R=500.0, C=1366e-6):
    """
    Generates the RC-filtered continuous Fourier series of a square wave.
    
    Args:
        T (float): Period of the square wave.
        num_periods (int): Number of periods to generate.
        num_harmonics (int): Number of odd harmonics to use.
        num_points_per_period (int): Number of time points per period.
        R (float): Resistance in Ohms.
        C (float): Capacitance in Farads.
        
    Returns:
        t (numpy.ndarray): Time vector.
        y_filtered (numpy.ndarray): Generated RC-filtered square wave amplitude.
    """
    omega_0 = 2 * np.pi / T
    total_time = num_periods * T
    total_points = num_periods * num_points_per_period
    t = np.linspace(0, total_time, total_points, endpoint=False)
    
    y_filtered = np.zeros_like(t)
    
    print("\n--- RC Filter Analytical Calculation ---")
    print(f"R = {R} Ohms, C = {C} Farads")
    print("-" * 60)
    
    for i in range(num_harmonics):
        n = 2 * i + 1
        omega_n = n * omega_0
        
        # Original harmonic amplitude for square wave
        A_n = 4 / (n * np.pi)
        
        # Transfer function H(omega) = 1 / (1 + j * omega * R * C)
        H = 1 / (1 + 1j * omega_n * R * C)
        
        # Amplitude factor |H(omega)| and phase shift angle(H(omega))
        amp_factor = np.abs(H)
        phase_shift = np.angle(H)
        
        print(f"Harmonic n={n:<2} | Amplitude Factor: {amp_factor:.4f} | Phase Shift: {phase_shift:.4f} rad")
        
        # Apply amplitude factor and phase shift to the harmonic
        y_filtered += A_n * amp_factor * np.sin(omega_n * t + phase_shift)
        
    print("-" * 60)
    return t, y_filtered

def add_noise_and_sabotage(t, y, T=66, start_period=70, end_period=75):
    """
    Adds Gaussian noise to the signal and injects a high-frequency voltage spike
    between the specified periods to simulate sabotage.
    
    Args:
        t (numpy.ndarray): Time vector.
        y (numpy.ndarray): Original signal array.
        T (float): Period of the signal.
        start_period (int): Period index where sabotage begins.
        end_period (int): Period index where sabotage ends.
        
    Returns:
        numpy.ndarray: The corrupted signal array.
    """
    # 1. Add random Gaussian noise
    noise_std_dev = 0.05  # Adjust noise level as needed
    y_noisy = y + np.random.normal(0, noise_std_dev, size=y.shape)
    
    # 2. Inject massive high-frequency voltage spike (sabotage)
    t_start = start_period * T
    t_end = end_period * T
    mask = (t >= t_start) & (t < t_end)
    
    # High frequency spike (e.g., 50x fundamental frequency)
    spike_freq = 50 * (2 * np.pi / T)
    spike_amplitude = 5.0  # Massive spike amplitude compared to the base signal
    spike = spike_amplitude * np.sin(spike_freq * t)
    
    y_corrupted = np.copy(y_noisy)
    y_corrupted[mask] += spike[mask]
    
    return y_corrupted

if __name__ == "__main__":
    # Parameters given in the task
    T = 66  # Based on the last two digits of student ID (4034366)
    num_periods = 100
    num_harmonics = 9
    
    # RC filter parameters
    R = 0.5 * 1000          # 0.5 kOhm
    C = (1000 + 366) * 1e-6 # 1366 uF
    
    # Generate the original data
    t, y = generate_square_wave_fourier(T=T, num_periods=num_periods, num_harmonics=num_harmonics)
    
    print("Successfully generated Fourier series of a square wave.")
    print(f"-> Period (T): {T}")
    print(f"-> Fundamental angular frequency (omega_0): {2 * np.pi / T:.4f} rad/s")
    print(f"-> Number of periods: {num_periods}")
    print(f"-> Number of odd harmonics used: {num_harmonics}")
    print(f"-> Original data shape: t={t.shape}, y={y.shape}")
    
    # Generate the RC-filtered data
    t_filtered, y_filtered = generate_rc_filtered_square_wave(
        T=T, num_periods=num_periods, num_harmonics=num_harmonics, R=R, C=C
    )
    
    print(f"-> Filtered data shape: t={t_filtered.shape}, y_filtered={y_filtered.shape}")
    
    # Add noise and sabotage
    y_corrupted = add_noise_and_sabotage(t_filtered, y_filtered, T=T, start_period=70, end_period=75)
    
    # Create data directory if it doesn't exist
    # Constructing path relative to this file's location to ensure it works anywhere
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    os.makedirs(data_dir, exist_ok=True)
    
    # Save the corrupted 1D array locally inside the data/ folder
    save_path = os.path.join(data_dir, "corrupted_signal.npy")
    np.save(save_path, y_corrupted)
    
    print("\n--- Sabotage Applied ---")
    print(f"-> Noise and sabotage successfully added (periods 70-75).")
    print(f"-> Corrupted signal saved to: {save_path}")
    print(f"-> Saved array shape: {y_corrupted.shape}")
    
    # -------------------------------------------------------------------------
    # Part 4: Plotting
    # -------------------------------------------------------------------------
    print("\n--- Generating Plots ---")
    
    # Define time windows based on periods
    # Normal window: periods 60 to 65
    mask_normal = (t_filtered >= 60 * T) & (t_filtered < 65 * T)
    t_normal = t_filtered[mask_normal]
    y_normal = y_corrupted[mask_normal]
    
    # Anomaly window: periods 70 to 75
    mask_anomaly = (t_filtered >= 70 * T) & (t_filtered < 75 * T)
    t_anomaly = t_filtered[mask_anomaly]
    y_anomaly = y_corrupted[mask_anomaly]
    
    # Create a figure with two subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Normal Noisy Signal Window
    ax1.plot(t_normal, y_normal, color='blue', linewidth=1)
    ax1.set_title("Normal Noisy Signal (Periods 60-65)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Anomaly Spike Window
    ax2.plot(t_anomaly, y_anomaly, color='red', linewidth=1)
    ax2.set_title("Anomaly Spike Window (Periods 70-75)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(data_dir, "data_feed.png")
    plt.savefig(plot_path, dpi=300)
    print(f"-> Plot successfully saved to: {plot_path}")
