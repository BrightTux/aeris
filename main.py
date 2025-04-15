# for llm and general usage
import requests
import configparser
import atexit

# voice control
import speech_recognition as sr
import pyaudio

# screen manipulation tools
import pygetwindow as gw
from screeninfo import get_monitors
import pyautogui

def process_command(command):
    payload = {
        "model": "mistral",
        "messages": [
            {"role": "system", "content": "You are a voice assistant that echos the input from the user. Do not say anything else."},
            {"role": "user", "content": command},
        ],
        "stream": False
    }

    try:
        response = requests.post(ollama_server_url, json=payload)
        response_data = response.json()
        return response_data["message"]["content"].strip().lower()
    except Exception as e:
        print(f"Error communicating with Ollama server: {e}")
        return None

# this is called from the background thread
def callback(recognizer, audio):
    try:
        command = recognizer_instance.recognize_faster_whisper(
                audio,
                language="english",
                model=config['voice']['model'],
                )

        print("Speech Recognition thinks you said " + command)
        process_command(command)

    except sr.UnknownValueError:
        print("Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Speech Recognition service; {0}".format(e))

def echo(string):
    print('running echo: ', string)

def process_voice_command():
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)

    print("Listening for command...")
    stop_listening = recognizer.listen_in_background(mic, callback)

@atexit.register()
def cleanup():
    # calling this function requests that the background listener stop listening
    stop_listening(wait_for_stop=False)

def listen_and_activate():
    recognizer = sr.Recognizer()

    with sr.Microphone(device_index=mic_index) as source:
        print("Listening for a command...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            command = recognizer.recognize_google(audio).lower()
            print(f"You said: {command}")

            # Send the command to the remote Ollama server
            action = process_command(command)
            if action:
                print(f"Assistant decided: {action}")
                if 'echo' in action:
                    echo('hello world')
                if 'amazing' in action:
                    echo('this is amazing')
                if 'funny' in action:
                    echo('this is funny')
            else:
                print("No valid response from Ollama.")

        except sr.UnknownValueError:
            print("Sorry, I couldn't understand.")
        except sr.RequestError:
            print("Speech recognition service error.")

if __name__ == '__main__':

    # first, determine which is the correct microphone.
    # TODO: find the name of the exact bluetooth mic that will be used.
    # TODO: add exception handling
    # TODO: currently the recognizer.recognize_google(audio) uses google STT function, change it to vosk or deepspeech for completely offline

    config = configparser.ConfigParser()
    config.read('config.ini')

    ollama_server_url = config['llm']['endpoint_test']
    model = whisper.load_model(config['voice']['model'])

    # for monitor identification
    for i, monitor in enumerate(get_monitors()):
        print(f"Monitor {i}: {monitor}")

    # List all available microphones
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"Microphone {index}: {name}")

    mic_index = input('Which mic would you like to use? :') or 8
    mic_index = int(mic_index)

    porcupine = pvporcupine.create(keywords=["Hello AERIS",])  # Choose your wake word

    # Setup PyAudio for microphone input
    pa = pyaudio.PyAudio()
    stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=porcupine.sample_rate,
            input=True,
            frames_per_buffer=porcupine.frame_length,
        )

    recognizer = sr.Recognizer()
    mic = sr.Microphone(device_index=mic_index)

    while True:
        pcm = stream.read(porcupine.frame_length)
        pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

        keyword_index = porcupine.process(pcm)
        if keyword_index >= 0:  # Wake word detected
            print("Wake word detected!")
            process_voice_command()

