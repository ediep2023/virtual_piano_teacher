import pyaudio
import wave
import time
import sys
import os


def record_me(filename):
    '''

    :param filename: name of output audio file
    :method: record audio from user for 5 seconds and saved it as a wav file in a directory. One
    :        number is being recorded for every sample because there is only one channel. The total number of data chunks = frames per second/ chunk size * numbers of seconds
    :        these chunks are stored and written out onto a file.
    '''
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1
    fs = 44100  # Record at 44100 samples per second
    seconds = 5
    try:
        time.sleep(1)
        p = pyaudio.PyAudio()  # Create an interface to PortAudio
        print('Recording')

        stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input=True)

        frames = []  # Initialize array to store frames

        # Store data in chunks for 5 seconds
        for i in range(0, int(fs / chunk * seconds)):
            data = stream.read(chunk)
            frames.append(data)

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        # Terminate the PortAudio interface
        p.terminate()

        print('Finished recording')

        # Save the recorded data as a WAV file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
        wf.close()
    except:
        pass

if __name__ == "__main__":
    record_me("/home/parallels/PycharmProjects/VirtualPianoTeacher2023/data/demo.wav")
