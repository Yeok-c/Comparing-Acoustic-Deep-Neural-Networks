import pyaudio
import wave
import datetime
import os

chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 1
fs = 48000  # Record at 44100 samples per second
seconds = 1
p = pyaudio.PyAudio()  # Create an interface to PortAudio
p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)

numdevices = info.get('deviceCount')
for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))

DEVICE_ID=1

print('Recording')

try:
    os.sys("mkdir ./test_save_files")
    try:
        os.sys("del /S ./test_save_files/**")    
    except:
        pass
except:
    pass


FILENAME = "./test_save_files/output"
WAV_LENGTH = 4

stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input=True,
                input_device_index = DEVICE_ID,)

frames = []  # Initialize array to store frames

# Store data in chunks for 3 seconds
frames_all = []
count = 0
loops=2000
for I in range(loops):
    frames = []
    while count < WAV_LENGTH: # First few runs, just extend don't delete
        for i in range(0, int(fs / chunk * seconds)):
            data = stream.read(chunk)
            frames.append(data)
        frames_all.extend(frames)
        count+=1

    # Number of iterations = (48000/1024) 'reads'/second * 1 (seconds)
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)
    del frames_all[:len(frames)]    
    frames_all.extend(frames)

    # Save the recorded data as a WAV file
    now = datetime.datetime.now()
    filename = "{}_{}.wav".format(FILENAME, now.strftime("%Y-%m-%d_%H-%M-%S-%f"))
    with wave.open(filename, 'wb') as wf:
        # wf = wave.open("{}_{}.wav".format(FILENAME, "%02d" % (I,)), 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames_all))
    wf.close()

# Stop and close the stream 
stream.stop_stream()
stream.close()
# Terminate the PortAudio interface
p.terminate()

print('Finished recording')