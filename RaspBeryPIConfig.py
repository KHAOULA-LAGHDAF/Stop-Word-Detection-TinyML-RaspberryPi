import sounddevice as sd
import numpy as np
import scipy.signal
import timeit
import python_speech_features
import RPi.GPIO as GPIO
from tflite_runtime.interpreter import Interpreter

# Parameters
debug_time = 1
debug_acc = 0
led_pin = 17  # Utiliser GPIO 17 pour les LED
word_threshold = 0.5
rec_duration = 0.5
window_stride = 0.5
sample_rate = 48000
resample_rate = 8000
num_channels = 1
num_mfcc = 16
model_path = 'wake_word_stop_lite.tflite'

# Sliding window for audio data
window = np.zeros(int(rec_duration * resample_rate) * 2)

# GPIO Configuration
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(led_pin, GPIO.OUT, initial=GPIO.LOW)

# Load model
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)

# Decimation function for downsampling
def decimate(signal, old_fs, new_fs):
    dec_factor = old_fs / new_fs
    if not dec_factor.is_integer():
        print("Error: target sample rate higher than original")
        return signal, old_fs
    return scipy.signal.decimate(signal, int(dec_factor)), new_fs

# Audio callback function
def sd_callback(rec, frames, time, status):
    GPIO.output(led_pin, GPIO.LOW)  # Turn off LED by default

    if status:
        print('Error:', status)

    rec = np.squeeze(rec)  # Remove the second dimension
    rec, new_fs = decimate(rec, sample_rate, resample_rate)  # Downsample

    # Update the sliding window
    window[:len(window)//2] = window[len(window)//2:]
    window[len(window)//2:] = rec

    # Calculate MFCCs
    mfccs = python_speech_features.base.mfcc(
        window, samplerate=new_fs, winlen=0.256, winstep=0.050,
        numcep=num_mfcc, nfilt=26, nfft=2048, preemph=0.0, ceplifter=0,
        appendEnergy=False, winfunc=np.hanning
    ).transpose()

    # Make prediction from model
    in_tensor = np.float32(mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1))
    interpreter.set_tensor(input_details[0]['index'], in_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    val = output_data[0][0]

    # Check if "stop" is detected
    if val > word_threshold:
        print('stop detected')
        GPIO.output(led_pin, GPIO.HIGH)  # Turn on the LED
        time.sleep(0.5)  # Keep LED on for a brief moment
        GPIO.output(led_pin, GPIO.LOW)  # Turn off the LED

    if debug_acc:
        print("Confidence:", val)

    if debug_time:
        print("Processing time:", timeit.default_timer() - start)

# Start microphone stream
print("Say 'stop' to test LED control...")
with sd.InputStream(channels=num_channels, samplerate=sample_rate,
                    blocksize=int(sample_rate * rec_duration), callback=sd_callback):
    while True:
        pass
