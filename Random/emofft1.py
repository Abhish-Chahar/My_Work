import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def generate_signal(freq, sampling_rate, duration):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t)
    return t, signal

def extract_features(signal, sampling_rate):
    n = len(signal)
    fft_freq = np.fft.fftfreq(n, d=1/sampling_rate)
    fft_signal = np.fft.fft(signal)
    freq_mask = (fft_freq >= 8) & (fft_freq <= 30)
    power_spectrum = np.abs(fft_signal[freq_mask]) ** 2
    return power_spectrum

def load_dataset():
    # Replace this function with your actual loading logic
    # For the sake of example, we'll generate random data
    np.random.seed(42)
    num_tasks = 8
    num_repetitions = 28
    num_channels = 45
    num_samples = 1200

    eeg_data = np.random.randn(num_tasks, num_repetitions, num_channels, num_samples)
    return eeg_data

def preprocess_dataset(eeg_data, sampling_rate):
    num_tasks, num_repetitions, num_channels, num_samples = eeg_data.shape

    X = np.zeros((num_tasks * num_repetitions, num_channels, 23))  # 23 frequency bins [8-30] Hz
    y = np.zeros(num_tasks * num_repetitions, dtype=int)

    for task in range(num_tasks):
        for repetition in range(num_repetitions):
            signal = eeg_data[task, repetition]
            for channel in range(num_channels):
                features = extract_features(signal[channel], sampling_rate)
                X[task * num_repetitions + repetition, channel] = features
                y[task * num_repetitions + repetition] = task

    return X, y

def plot_signal(t, signal):
    plt.figure(figsize=(10, 4))
    plt.plot(t, signal)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Time Domain Signal')
    plt.grid()
    plt.show()

def plot_frequency_domain(t, signal, sampling_rate):
    n = len(signal)
    fft_freq = np.fft.fftfreq(n, d=1/sampling_rate)
    fft_signal = np.fft.fft(signal)

    plt.figure(figsize=(10, 4))
    plt.plot(fft_freq, np.abs(fft_signal))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Frequency Domain Signal')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Signal parameters
    frequency = 5.0  # Hz
    sampling_rate = 600.0  # Hz
    duration = 2.0  # seconds

    # Generate signal
    t, signal = generate_signal(frequency, sampling_rate, duration)

    # Plot time domain signal
    plot_signal(t, signal)

    # Plot frequency domain signal
    plot_frequency_domain(t, signal, sampling_rate)

    # Load and preprocess dataset
    eeg_data = load_dataset()
    X, y = preprocess_dataset(eeg_data, sampling_rate)

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1, X.shape[-1]), y, test_size=0.2, random_state=42)

    # SVM classifier
    svm_classifier = SVC(kernel='linear', C=1.0)

    # Train the classifier
    svm_classifier.fit(X_train, y_train)

    # Test the classifier
    y_pred = svm_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
