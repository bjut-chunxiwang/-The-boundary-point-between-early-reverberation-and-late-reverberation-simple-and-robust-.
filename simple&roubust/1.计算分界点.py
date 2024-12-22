import pandas as pd
import librosa
import numpy as np
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

def process_files(csv_path, N=1024, preDelay=0):
    # Read the CSV file containing the paths to the .wav files
    df = pd.read_csv(csv_path)

    # Check if 'file_rir' column exists
    if 'file_rir' not in df.columns:
        raise ValueError("'file_rir' column not found in the CSV file.")

    # Prepare a list to store the t_abel values
    t_abels = []

    # Process each file listed in the CSV
    for index, row in df.iterrows():
        audio_file = row['file_rir']
        try:
            # Load the audio file
            IR, fs = librosa.load(audio_file, sr=None)  # Load audio with original sample rate

            # Compute the mixing time (t_abel) and echo density
            t_abel, _ = compute_echo(IR, N, fs, preDelay)
            t_abels.append(round(t_abel, 2))  # Round to 2 decimal places
            print(round(t_abel, 2))

        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            t_abels.append(np.nan)  # If there's an error, append NaN

    # Add the computed t_abel values to the DataFrame
    df['t_abel'] = t_abels

    # Save the updated DataFrame back to the CSV file
    df.to_csv(csv_path, index=False)
    print(f"Updated CSV file saved to {csv_path}")

# Usage
csv_path = r'C:\Users\Lenovo\Desktop\simple&roubust\read(BP).csv'  # Replace with the path to your CSV file
process_files(csv_path)
