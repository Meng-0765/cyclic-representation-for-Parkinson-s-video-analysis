import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.signal import find_peaks
from bisect import bisect_left


# ---------- Helpers ----------
def longest_monotone_subsequence(seq):
    """Return the longest monotone subsequence length."""
    def lis_length(arr):  # Longest increasing subsequence.
        sub = []
        for x in arr:
            pos = bisect_left(sub, x)
            if pos == len(sub):
                sub.append(x)
            else:
                sub[pos] = x
        return len(sub)
    return max(lis_length(seq), lis_length(seq[::-1]))


def extract_features(signal, fs=30, plot=False, save_path=None, file_name=None, peak_params=None):
    signal = np.ravel(signal)  # Ensure a one-dimensional signal.
    T = len(signal)

    # ---- 1. peak & trough detection ----
    if peak_params is None:
        peak_params = {}  # Empty parameters reproduce the raw find_peaks call.

    peaks, _ = find_peaks(signal, **peak_params)
    troughs, _ = find_peaks(-signal, **peak_params)

    # Period and frequency.
    if len(peaks) > 1:
        periods = np.diff(peaks) / fs
        freqs = 1.0 / periods
    else:
        periods, freqs = np.array([]), np.array([])

    # ---- 2. velocity, acceleration, jerk ----
    vel = np.abs(np.diff(signal, prepend=signal[0]))  # mean absolute first difference
    acc = np.abs(np.diff(vel, prepend=vel[0]))
    jerk = np.abs(np.diff(acc, prepend=acc[0]))

    # ---- 3. Features ----
    features = {}

    # frequency-related
    features["MeanFreq"] = np.mean(freqs) if len(freqs) > 0 else 0
    features["CovarFreq"] = np.std(freqs) / np.mean(freqs) if len(freqs) > 0 and np.mean(freqs)!=0 else 0

    # velocity-related
    features["MeanVel"] = np.mean(vel)
    features["CovarVel"] = np.std(vel) / np.mean(vel) if np.mean(vel)!=0 else 0

    # amplitude-related
    peak_vals = signal[peaks] if len(peaks) > 0 else np.array([0])
    features["MeanAmp"] = np.mean(peak_vals)
    features["CovarAmp"] = np.std(peak_vals) / np.mean(peak_vals) if np.mean(peak_vals)!=0 else 0

    # period range
    features["PeriodRange"] = np.max(periods) - np.min(periods) if len(periods) > 0 else 0

    # PrcInv
    if T > 1:
        lms = longest_monotone_subsequence(signal)
        features["PrcInv"] = (T - lms) / T
    else:
        features["PrcInv"] = 0

    # roughness
    valid_idx = acc > 1e-6
    ratio = jerk[valid_idx] / acc[valid_idx] if np.any(valid_idx) else [0]
    features["Roughness"] = np.median(ratio)

    # trend in amplitude
    if len(peak_vals) >= 3:
        n = len(peak_vals)
        AT1 = np.mean(peak_vals[:n//3])
        AT3 = np.mean(peak_vals[-n//3:])
        features["DiffAmp"] = (AT3 - AT1) / AT1 if AT1 != 0 else 0
    else:
        features["DiffAmp"] = 0

    # trend in velocity
    if len(vel) >= 3:
        n = len(vel)
        WT1 = np.mean(vel[:n//3])
        WT3 = np.mean(vel[-n//3:])
        features["DiffVel"] = (WT3 - WT1) / WT1 if WT1 != 0 else 0
    else:
        features["DiffVel"] = 0

    # ---- 4. Optional visualization ----
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(signal, label="PCA compressed signal", color="blue")
        if len(peaks) > 0:
            plt.plot(peaks, signal[peaks], "ro", label="Peaks")
        if len(troughs) > 0:
            plt.plot(troughs, signal[troughs], "go", label="Troughs")
        plt.title(f"PCA Signal with Peaks/Troughs ({file_name})")
        plt.xlabel("Time (frames)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(os.path.join(save_path, f"{file_name}.png"))
            plt.close()
        else:
            plt.show()

    return features


# ---------- Folder processing ----------
def process_folder(input_folder, output_csv, fs=30, plot_folder=None, peak_params=None):
    results = []
    file_names = []

    if plot_folder:
        os.makedirs(plot_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".npy"):
            file_path = os.path.join(input_folder, file_name)

            # Read the embedding array (T x embedding_dim).
            data = np.load(file_path)

            # Compress to one dimension with PCA.
            pca = PCA(n_components=1)
            compressed = pca.fit_transform(data)

            # Extract features and optionally plot.
            feats = extract_features(
                compressed,
                fs=fs,
                plot=bool(plot_folder),
                save_path=plot_folder,
                file_name=os.path.splitext(file_name)[0],
                peak_params=peak_params
            )
            results.append(feats)
            file_names.append(file_name)

            print(f"Processed: {file_name}")

    # Convert to a DataFrame.
    df = pd.DataFrame(results, index=file_names)
    df.to_csv(output_csv)
    print(f"Saved all results to {output_csv}")


# ---------- Example ----------
if __name__ == "__main__":
    input_folder = "./results/embeddings_whole"   # Replace with an embedding folder.
    output_csv = "features_finger.csv"
    plot_folder = "plots"               # Output directory for plots.

    # Optional peak-detection parameters.
    peak_params = {
        "distance": 10,      # Minimum distance between peaks, in frames.
        # "prominence": 0.2, # Minimum peak prominence.
        "width": 3           # Minimum peak width.
    }

    process_folder(input_folder, output_csv, fs=30, plot_folder=None, peak_params=peak_params)
