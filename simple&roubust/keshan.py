import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import hann
from scipy.special import erfc

def compute_echo(IR, N, fs, preDelay):
    # Mixing threshold for echo density
    mixingThresh = 1.0

    # Initialize arrays
    s = np.zeros(len(IR))
    echo_dens = np.zeros(len(IR))

    # Create Hann window and normalize it
    wTau = hann(N)
    wTau = wTau / np.sum(wTau)

    halfWin = N // 2  # Half window size

    if len(IR) < N:
        raise ValueError(
            "IR shorter than analysis window length (1024 samples). Provide at least an IR of some 100 ms.")

    # Calculate the total energy of the IR
    total_energy = np.sum(IR ** 2)

    # Indices for sparse calculation (to speed up the process)
    sparseInd = np.arange(0, len(IR), 500)

    for n in sparseInd:
        if n <= halfWin:
            hTau = IR[0:n + halfWin]
            wT = wTau[-(halfWin + n):]
        elif halfWin < n < len(IR) - halfWin:
            hTau = IR[n - halfWin:n + halfWin]
            wT = wTau
        elif n >= len(IR) - halfWin:
            hTau = IR[n - halfWin:]
            wT = wTau[:len(hTau)]
        else:
            raise ValueError("Invalid n condition")

        # Calculate the standard deviation of the windowed signal
        s[n] = np.sqrt(np.sum(wT * (hTau ** 2)))

        # Count samples where the signal is greater than the standard deviation
        tipCt = np.abs(hTau) > s[n]

        # Echo density calculation
        echo_dens[n] = np.sum(wT * tipCt)

    # Normalize echo density
    echo_dens = echo_dens / erfc(1 / np.sqrt(2))

    # Interpolation to match the length of the input signal
    echo_dens = np.interp(np.arange(len(IR)), sparseInd, echo_dens[sparseInd])

    # Find the first point where echo density exceeds the threshold and energy is above 10% of total energy
    energy_condition = np.cumsum(IR ** 2) / total_energy >= 0.1
    d = np.where((echo_dens > mixingThresh) & energy_condition)[0]

    # Calculate t_abel (mixing time)
    if len(d) > 0:
        t_abel = (d[0] - preDelay) / fs * 1000  # Convert to milliseconds
    else:
        t_abel = 0
        print("Mixing time not found within the given limits.")
    print(t_abel)

    return t_abel, echo_dens

def plot_results(IR, echo_dens, fs, t_abel):
    # Time axis for plotting
    t = np.arange(len(IR)) / fs

    # Plotting the IR and Echo Density
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    # Plot the Impulse Response
    axs[0].plot(t, IR, label="Impulse Response", color='b')
    axs[0].set_title("Impulse Response")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True)

    # Mark the mixing time
    axs[0].axvline(t_abel / 1000, color='r', linestyle='--', label=f"Mixing Time (t_abel): {t_abel:.2f} ms")
    axs[0].legend()

    # Plot the Echo Density
    axs[1].plot(t, echo_dens, label="Echo Density", color='g')
    axs[1].set_title("Echo Density")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Echo Density")
    axs[1].grid(True)

    # Display the figure
    plt.tight_layout()
    plt.show()

# Load the audio file
audio_file = "impulseresponseheslingtonchurch-001.wav"  # Replace with your audio file path
IR, fs = librosa.load(audio_file, sr=None)  # Load audio signal with original sampling rate

# Define parameters
N = 1024  # Window length (must be even)
preDelay = 0 # Pre-delay in samples, you can adjust this based on your needs

# Compute the mixing time (t_abel) and echo density
t_abel, echo_dens = compute_echo(IR, N, fs, preDelay)

# Plot the results
plot_results(IR, echo_dens, fs, t_abel)
