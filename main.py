#!/usr/bin/python

import ast
import atexit
import configparser
import json
import os
import platform
import subprocess

# voice control
import pyaudio
import pyttsx3
import speech_recognition as sr

# LLM
import dspy

from screeninfo import get_monitors
from typing import Literal, Callable

EXEC_PLATFORM = platform.system()
if EXEC_PLATFORM == "Windows":
    import pygetwindow as gw
    import pyautogui

# similar sounding wake phrases
wake_phrases = [
    "hello aeries",
    "hello aries",
    "hello iris",
    "hello aeris",
]


def play_video(video_name, screen_number):
    print("Running play_video", video_name, screen_number)
    # raise NotImplementedError


def pause_video(screen_number):
    print("Running pause_video", screen_number)
    # raise NotImplementedError


def stop_video(screen_number):
    print("Running stop_video", screen_number)
    # raise NotImplementedError


def control_presentation(presentation_name, screen_number):
    print("Running control_presentation", presentation_name, screen_number)
    # raise NotImplementedError


def turn_on_experience_wall():
    print(
        "Running turn_on_experience_wall",
    )
    # raise NotImplementedError


def set_volume(volume_level):
    print("Running set_volume", volume_level)
    # raise NotImplementedError


def split_screen(screen_number, content_left, content_right):
    print("Running split_screen", content_left, content_right)
    # raise NotImplementedError


def system_sleep():
    print(
        "Running system_sleep",
    )
    # raise NotImplementedError


def execute_command(command_str: str):
    try:
        # Safely parse the command string
        tree = ast.parse(command_str, mode="eval")
        if not isinstance(tree.body, ast.Call):
            raise ValueError("Command must be a function call")

        func_name = tree.body.func.id
        args = {}
        for kw in tree.body.keywords:
            args[kw.arg] = ast.literal_eval(kw.value)

        if func_name not in COMMAND_REGISTRY:
            raise ValueError(f"Unknown command: {func_name}")

        COMMAND_REGISTRY[func_name](**args)

    except Exception as e:
        print(f"Failed to execute command: {e}")


def say(text):
    """
    TTS function, currently a dummy since i can't get it working
    """
    print(text)
    # tts_engine.say(text)
    # tts_engine.runAndWait()
    # tts_engine.stop()


class Categorize(dspy.Signature):
    """Classify the intent of the command."""

    command: str = dspy.InputField()
    intent: Literal[
        "screen control (on/off)",
        "video content control",
        "powerpoint slide content control",
        "system rest/aeris go to sleep",
        "experience wall control (on/off/content)",
        "content control on all screensvolume control",
        "split screen control",
    ] = dspy.OutputField()
    confidence: float = dspy.OutputField()


class CommandControl(dspy.Signature):
    """Determine the command details based on the input request and intention.

    Input:
    - intention: intention of the request
    - request: actual request from the user
    Output:
    # Here are the valid responses:
    - play_video(video_name, screen_number)
    - pause_video(screen_number)
    - stop_video(screen_number)
    - control_presentation(presentation_name, screen_number)
    - turn_on_experience_wall()
    - set_volume(volume_level)
    - split_screen(screen_number, content_left, content_right)
    - system_sleep()

    """

    intention: str = dspy.InputField()
    request: str = dspy.InputField()

    command: str = dspy.OutputField(desc="Actual function and parameters to run")
    confidence: float = dspy.OutputField(
        desc="For downstream error handling or routing"
    )  # For downstream error handling or routing


def process_command(command):
    intent_res = classify(command=command)
    print(
        f"Received: {command}. Predicted: {intent_res.intent}, {intent_res.confidence}"
    )

    # now, based on the intent of the command, determine what commands to run.
    command_res = get_command_actions(
        intention=intent_res.intent,
        request=command,
    )

    if command_res.confidence <= float(config["voice_control"]["confidence"]):
        return "Failed to process command, low confidence"

    print(f"Generated command: {command_res}")
    # if it passes the conf threshold
    execute_command(command_res.command)


def play_video_on_monitor(monitor_index, video_path):
    # Use VLC or another player; this is a simple VLC command example
    monitors = get_monitors()
    if monitor_index >= len(monitors):
        raise ValueError("Invalid monitor index")

    x = monitors[monitor_index].x
    y = monitors[monitor_index].y
    command = f'vlc --qt-start-minimized --video-x={x} --video-y={y} "{video_path}"'
    subprocess.Popen(command, shell=True)


def adjust_volume(action, value=None):
    if action == "increase":
        os.system("amixer set Master 5%+")
    elif action == "decrease":
        os.system("amixer set Master 5%-")
    elif action == "mute":
        os.system("amixer set Master mute")
    elif action == "unmute":
        os.system("amixer set Master unmute")


def control_screen(action, monitor_index=0):
    # Placeholder — implementation depends on OS
    if action == "turn_off":
        print(f"Turning off screen {monitor_index}")
    elif action == "turn_on":
        print(f"Turning on screen {monitor_index}")


def echo(string):
    print("running echo: ", string)


def listen_and_activate_whisper():
    recognizer = sr.Recognizer()

    with sr.Microphone(device_index=mic_index) as source:
        print("Calibrating for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print("Always listening for wake word...")

        while True:
            try:
                audio = recognizer.listen(source)  # no timeout or phrase_time_limit
                transcript = recognizer.recognize_faster_whisper(audio)
                print(f"You said: {transcript}")

                if any(w.lower() in transcript.lower() for w in wake_phrases):
                    print("Wake word detected. Listening for command...")

                    # Optional: short pause to give time for user to think
                    print("You may speak your command now.")
                    audio = recognizer.listen(source)
                    command = recognizer.recognize_faster_whisper(audio).lower()
                    print(f"Command received: {command}")

                    action = process_command(command)
                    if action:
                        print(f"Assistant decided: {action}")
                        if "echo" in action:
                            echo("hello world")
                        elif "amazing" in action:
                            echo("this is amazing")
                        elif "funny" in action:
                            echo("this is funny")
                    else:
                        print("No valid response from Ollama.")
            except sr.UnknownValueError:
                print("Didn't catch that.")
            except sr.RequestError as e:
                print("Recognition error:", e)
            except Exception as e:
                print("Unhandled error:", e)


if __name__ == "__main__":
    # determine what system are we running on:
    print(EXEC_PLATFORM)

    # TODO: find the name of the exact bluetooth mic that will be used.
    # TODO: add exception handling

    config = configparser.ConfigParser()
    config.read("config.ini")

    # DSPY related
    ollama_server_url = config["llm"]["endpoint_test"]
    lm = dspy.LM(config["llm"]["model"], api_base=ollama_server_url, api_key="")
    dspy.configure(lm=lm)
    classify = dspy.Predict(Categorize)
    get_command_actions = dspy.ChainOfThought(CommandControl)

    # dictionary of video files
    video_dict = config["presentation"]

    COMMAND_REGISTRY: dict[str, Callable] = {
        "play_video": play_video,
        "pause_video": pause_video,
        "stop_video": stop_video,
        "control_presentation": control_presentation,
        "turn_on_experience_wall": turn_on_experience_wall,
        "set_volume": set_volume,
        "split_screen": split_screen,
        "system_sleep": system_sleep,
    }

    with open("./voice_assistant_test_cases.json", "r") as f:
        test_cases = json.load(f)

    for tc in test_cases:
        print("input: ", tc)
        process_command(tc["command"])

    # tts_engine = pyttsx3.init()
    # voices = tts_engine.getProperty('voices')       #getting details of current voice
    # tts_engine.setProperty('voice', voices[0].id)  #changing index, changes voices. o for male
    # tts_engine.setProperty('voice', voices[1].id)   #changing index, changes voices. 1 for female
    #
    # for monitor identification
    for i, monitor in enumerate(get_monitors()):
        print(f"Monitor {i}: {monitor}")

    # List all available microphones
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"Microphone {index}: {name}")

    mic_index = input("Which mic would you like to use? :") or 8
    mic_index = int(mic_index)

    listen_and_activate_whisper()
