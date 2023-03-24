from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput

import pyaudio
import wave
import time
import sys
import os

from multilabel_wav2score_jingles import wav2score as wav2score
from music21_vis import generate_music_sheet as generate_music_sheet

class MusicLesson(App):
    def build(self):
        self.window = GridLayout()
        self.window.cols = 1
        self.window = GridLayout()
        self.window.cols = 1
        self.window.size_hint = (0.9, 0.9)
        self.window.pos_hint = {"center_x": 0.5, "center_y":0.5}
        self.greeting1 = Label(text="Virtual Piano Teacher - Personalized Music Lesson", font_size='20sp')
        self.window.add_widget(self.greeting1)
        self.button = Button(
                      text= "Record",
                      size_hint= (1,0.5),
                      bold= True,
                      background_color ='#00FFCE'
                      )
        self.button.bind(on_press=self.recorder_callback)
        self.window.add_widget(self.button)

        self.button = Button(
                      text= "Analyze",
                      size_hint= (1,0.5),
                      bold= True,
                      background_color ='#33B5FF'
                      )
        self.button.bind(on_press=self.analysis_callback)
        self.window.add_widget(self.button)
        return self.window

    def analysis_callback(self, instance):
        wav2score()
        JINGLE1_PREDICTION_FOLDER = '/home/parallels/PycharmProjects/VirtualPianoTeacher2023/data/StudentMultiLabelPrediction'
        title = "Jingle 1"
        summary = 'Emily Diep\nred - wrong note, blue - extra note, black - correct note, orange - wrong tempo'
        generate_music_sheet(JINGLE1_PREDICTION_FOLDER, title, summary)

    def recorder_callback(self, instance):
        filename = "/home/parallels/PycharmProjects/VirtualPianoTeacher2023/data/JingleOne_demo.wav"
        chunk = 1024  # Record in chunks of 1024 samples
        sample_format = pyaudio.paInt16  # 16 bits per sample
        channels = 1
        fs = 44100  # Record at 44100 samples per second
        seconds = 6
        try:
            p = pyaudio.PyAudio()  # Create an interface to PortAudio
            stream = p.open(format=sample_format,
                            channels=channels,
                            rate=fs,
                            frames_per_buffer=chunk,
                            input=True)

            frames = []  # Initialize array to store frames

            # Store data in chunks
            for i in range(0, int(fs / chunk * seconds)):
                data = stream.read(chunk)
                frames.append(data)

            # Stop and close the stream
            stream.stop_stream()
            stream.close()
            # Terminate the PortAudio interface
            p.terminate()

            self.greeting1.text = "Finished Recording"

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
    MusicLesson().run()


