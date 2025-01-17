# 🗣️ TinyML-Based Voice Command Recognition for Real-Time LED Activation on Raspberry Pi

This project provides a hands-on introduction to TinyML, demonstrating how to build a real-time voice command recognition system that controls an LED on a Raspberry Pi. We use a Convolutional Neural Network (CNN) to identify the "stop" keyword and trigger LED activation, illustrating essential TinyML deployment steps, including model training, optimization, and real-time implementation.

## 📜 Project Overview

This project demonstrates the implementation of a TinyML voice command recognition system that:

*   Recognizes the keyword "**stop**".
*   Activates an LED on a Raspberry Pi upon detection of the "stop" command.
*   Leverages a CNN model trained on MFCC features from audio signals.
*   Employs TensorFlow Lite for model optimization and Raspberry Pi deployment.

## 🌟 Key Features

*   **MFCC Feature Extraction**: Extracts relevant features from audio recordings using Mel-Frequency Cepstral Coefficients.
*   **CNN Model**: Classifies audio data as "stop" or "not stop" using a Convolutional Neural Network.
*   **TensorFlow Lite Optimization**: Converts the model for efficient deployment on edge devices like Raspberry Pi.
*   **Real-Time Deployment**: Implements the model on a Raspberry Pi for real-time keyword detection and LED control.

## 📁 Project Files

The repository includes the following files:

*   **`mfcc_extraction.py`**: Extracts MFCC features from audio samples.
*   **`mfcc_classifier.py`**: Trains a CNN model using MFCC features to classify audio as "stop" or "not stop".
*   **`model_optimization_tflite.py`**: Converts the trained model to TensorFlow Lite format.
*   **`raspberry_pi_inference.py`**: Loads the TensorFlow Lite model and controls the LED on a Raspberry Pi using GPIO.

## ⚙️ Project Pipeline

1.  **Feature Extraction with MFCC:**
    *   The `mfcc_extraction.py` script extracts MFCC features from audio samples.

2.  **CNN Classifier for Keyword Detection:**
    *   The `mfcc_classifier.py` script trains a CNN model to classify audio data using the extracted MFCC features.

3.  **Model Optimization:**
    *   The `model_optimization_tflite.py` script converts the trained model to a TensorFlow Lite format.

4.  **Deployment on Raspberry Pi:**
    *   The `raspberry_pi_inference.py` script deploys the TensorFlow Lite model on a Raspberry Pi, controlling an LED based on the keyword detection.

## 🚀 Installation

To set up the project:

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/KHAOULA-LAGHDAF/Stop-Word-Detection-TinyML-RaspberryPi.git
    cd TinyML-Voice-Command-LED
    ```


## ⚙️ Requirements

*   **Hardware:**
    *   Raspberry Pi (any model with GPIO support)

*   **Software:**
    *   Python 3.7+
    *   TensorFlow (for training and converting to TensorFlow Lite)
    *   Raspbian OS on Raspberry Pi
    *   RPi.GPIO (for Raspberry Pi GPIO control)
    *   SoundDevice or similar (for audio input)

## 🎥 Demonstration Video

A demonstration video showcasing the working model on the Raspberry Pi (where the LED lights up in response to the voice command).
