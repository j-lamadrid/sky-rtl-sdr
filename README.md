# RTL-SDR Radio Astronomy: Meteor Scatter Identification & Classification

## Hardware Architecture

* **Input:** Dipole Antenna (tuned to ~50-144 MHz) → **RTL-SDR v3 Dongle**
* **Host Processor:** **Raspberry Pi 4/5**
  * *Role:* Raw Signal Acquisition, FFT Processing, Noise Floor Estimation
* **Inference Edge:** **Arduino Nano 33 BLE Sense**
  * *Role:* Gaussian Mixture Model (GMM) Classification
* **Output:** OLED/LED Display (Visual Alert) + CSV Logs on Pi USB Drive (Data Archival)
  * *Note:* Full `matplotlib` visualization including star, planet, and satellite positions already built in test.py (may be included on native display)

## Software Logic

### 1. Signal Processing Stage (Raspberry Pi | Python)

* **Signal Acquisition:** Reads raw I/Q samples from the RTL-SDR using `pyrtlsdr`
* **Transformation:** Performs a Fast Fourier Transform (FFT) to generate the Power Spectral Density (PSD)
* **Denoising (Kalman Filter):** Runs a **1D Scalar Kalman Filter** to continuously estimate and track the dynamic noise floor
* **Segmentation:** Detects anomalies by identifying spikes >3σ above the Kalman noise floor
* **Feature Extraction:** Calculates physics-based metrics for the candidate signal:
  * `SNR` (Peak Signal-to-Noise Ratio)
  * `Duration` (Time above threshold)
  * `Rise_Time` (Slope of signal onset)

### 2. Transmission (UART)
* The Pi serializes the features into a lightweight string and transmits it to the microcontroller via USB Serial

### 3. Inference Stage (Arduino Nano 33 BLE Sense | C++)
The Arduino performs classification using parameters learned from offline training on BRAMS dataset

* **Model:** **Gaussian Mixture Model (GMM)** (Learned clusters: *Noise*, *Meteor*, *Anomaly*)
* **Classification:** Calculates the **Mahalanobis Distance** between the incoming vector and the learned cluster means
* **Decision:**
  * Computes the Bayesian Evidence Score (Posterior Probability)
  * **IF** P(Meteor) > 90%: Trigger **METEOR**
  * **IF** P(Anomaly) > 90%: Reject as interference

## Code Requirements

* **Python 3.x:** `pyrtlsdr`, `numpy`, `scipy` (Signal Processing), `scikit-learn` (Model Training)
* **C++ (Arduino):** Real-time inference engine, Matrix Math (custom implementation for Mahalanobis distance)
* **Hardware Libraries:** `Wire.h` (OLED), `Serial` (UART Communication)
