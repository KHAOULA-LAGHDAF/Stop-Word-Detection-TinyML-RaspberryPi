# Stop-Word-Detection-TinyML-RaspberryPi
TinyML-Based Voice Command Recognition System for Real-Time Activation of LED Indicator on Raspberry Pi


This project is a voice recognition system that detects the word "stop" and activates a red LED lamp. It utilizes TinyML with TensorFlow Lite and runs on a Raspberry Pi, allowing for efficient real-time recognition and response on a small device. The system uses a pre-trained model, converted and optimized with TensorFlow Lite, for wake-word detection based on the "stop" command.


Project Overview
Goal: Develop a real-time voice recognition model that activates an LED when the word "stop" is detected.
Hardware: Raspberry Pi with an LED lamp connected.
Software Frameworks: TensorFlow, TensorFlow Lite, and TinyML principles.
Data: Google Speech Commands Dataset.
Feature Extraction: Mel Frequency Cepstral Coefficients (MFCC).
Model Architecture: Convolutional Neural Network (CNN).


System Workflow
1. Data Preparation: The Google Speech Commands Dataset was used for training. This dataset contains various short audio clips, including the target word "stop."

2. Feature Extraction: We extracted features using the MFCC (Mel Frequency Cepstral Coefficients) technique, which is a popular approach in audio signal processing. MFCC helps to capture essential features from the audio signals that contribute to distinguishing different words.

3. Model Training: A CNN (Convolutional Neural Network) was used to train the model for wake-word detection. The CNN architecture allows the model to learn and recognize the characteristics of the "stop" word, filtering out other sounds and background noise.

4. Model Conversion: After training, the model was converted to TensorFlow Lite format using TensorFlow Lite, which optimizes it for deployment on edge devices like the Raspberry Pi. This process involves reducing the model's size and complexity while retaining accuracy.

5. Deployment on Raspberry Pi: The optimized model was deployed on a Raspberry Pi, where it continuously listens for the "stop" command. When the system recognizes the word, it triggers a GPIO signal that turns on a red LED lamp.


TinyML opens up numerous possibilities in fields where low latency and low power consumption are essential, such as wearable devices, IoT applications, and smart home automation. Through this project, I gained hands-on experience with the TinyML workflow, including data preprocessing, feature extraction with MFCC, model training, TensorFlow Lite conversion, and deployment on embedded hardware. These skills are invaluable for developing AI-driven solutions on resource-constrained devices and have expanded my understanding of practical machine learning deployment.

By completing this project, I not only learned to build a fully functional TinyML application but also deepened my knowledge in voice recognition and embedded systems. This experience has been a significant step in my journey to specialize in edge computing and AI, equipping me with practical insights for future TinyML and IoT projects.









