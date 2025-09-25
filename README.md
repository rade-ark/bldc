## Dataset Overview

* **Title**: Dataset of audio signals from brushless DC motors for predictive maintenance
* **Source**: Universidad del Cauca, Colombia
* **Publication**: *Data in Brief* (2023)
* **Format**: 43 raw `.wav` files (\~10s each, sampled at 16 kHz)

---

## Content & Structure

* **Motors tested**: 4 × A2212 BLDC motors
* **Conditions recorded**:

  * Healthy motors
  * Propeller fault
  * Bearing fault
* **Organization**: Audio files grouped by motor (M1, M2, M3, M4) and fault type.
* **File naming**: Indicates motor speed in RPM (e.g., `1700.wav`, `1850.wav`).

Each recording:

* Duration: \~10 seconds
* Sampling rate: 16 kHz
* Size: \~160,000 samples

---

## Experimental Setup

* **Power source**: 11.1 V, 2100 mAh LiPo battery
* **Control**: ESP32 microcontroller + 30A ESC (speed control via potentiometer or Blynk IoT app)
* **Recording**: MCJR-005 capacitor microphone + Audacity software
* **Environment**: Controlled, minimizing background noise

---

## Purpose & Value

* Designed for **Predictive Maintenance (PdM)** in Industry 4.0
* Useful for:

  * Signal processing experiments
  * Feature extraction (statistical & spectral features)
  * Machine learning models (classification/regression) for fault detection

---

## Limitations

* Collected in controlled environments (little environmental noise).
* Only includes **3 conditions** (healthy, bearing fault, propeller fault).

---

## Ethics & Access

* No humans/animals involved.
* Open Access under CC-BY 4.0.
* Available on **Mendeley Data**: [https://data.mendeley.com/datasets/j4yr5fmhv4/1](https://data.mendeley.com/datasets/j4yr5fmhv4/1)

---
