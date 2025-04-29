import ast
import atexit
import configparser
import cv2
import comtypes.client
import dspy
import io
import json
import logging
import numpy as np
import os
import platform
import pyaudio
import pyttsx3
import speech_recognition as sr
import subprocess
import tempfile
import threading
import time
import string

from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template, redirect, url_for
from collections import deque
from pathlib import Path
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from screeninfo import get_monitors
from tkinter import Tk, filedialog
from typing import Literal, Callable

EXEC_PLATFORM = platform.system()
if EXEC_PLATFORM == "Windows":
    import pygetwindow as gw
    import pyautogui

    # audio/volume control
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    from ctypes import cast, POINTER

app = Flask(__name__)

# Width ratios for each panel (default: equal)
width_ratios = [1, 1, 1]  # Can be adjusted dynamically if needed
canvas_width = 3064
canvas_height = 672
canvas_width = 800
canvas_height = 200

# In-memory log storage
log_storage = deque(maxlen=20)
log_lock = threading.Lock()

# Configure logging
logging.basicConfig(level=logging.INFO)
voice_logger = logging.getLogger("voice_assistant")

def format_log_message(level, message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")  # Format the timestamp
    return f"[{timestamp}] [{level}] *{message}"

def log_to_memory(message, level='INFO'):
    formatted_message = format_log_message(level, message)
    with log_lock:
        try:
            previous_msg = log_storage[-1].split('*')[-1]
            if previous_msg == message:
                return
        except IndexError:
            pass  # when launched, the deque is empty
        log_storage.append(formatted_message)

def write_logs_to_file():
    log_file = "voice_assistant.log"
    while True:
        with log_lock:
            with open(log_file, 'a', encoding='utf-8', errors='ignore') as f:
                for line in log_storage:
                    f.write(line + "\n")
        time.sleep(60)  # Write logs to file every 60 seconds

log_thread = threading.Thread(target=write_logs_to_file, daemon=True)
log_thread.start()

class MediaPanel:
    def __init__(self, panel_id):
        self.panel_id = panel_id
        self.filepath = ""
        self.cap = None
        self.playing = False
        self.video_paused = False
        self.frame = np.zeros((canvas_height, 1, 4), dtype=np.uint8)
        self.width = None
        self.lock = threading.Lock()

        self.slides = []
        self.slide_index = 0
        self.slide_duration = 5
        self.last_slide_time = time.time()

    def set_video(self, path):
        with self.lock:
            self.filepath = path
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(path)
            self.playing = False

    def play(self):
        with self.lock:
            self.playing = True

    def pause(self):
        with self.lock:
            self.playing = False
            self.video_paused = True

    def stop(self):
        with self.lock:
            self.playing = False
            width = self.width or 1
            self.frame = np.zeros((canvas_height, width, 4), dtype=np.uint8)
            if self.cap:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def clear_slides(self):
        with self.lock:
            self.slides = []

    def update_frame(self, width):
        with self.lock:
            if self.slides:
                now = time.time()
                if now - self.last_slide_time > self.slide_duration:
                    self.slide_index = (self.slide_index + 1) % len(self.slides)
                    self.last_slide_time = now
                img = cv2.imread(self.slides[self.slide_index])
                rgba_image = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                self.frame = cv2.resize(rgba_image, (width, canvas_height))

            elif self.playing and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    # if finish playing, loop it
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                if ret:
                    rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                    rgba_frame = cv2.resize(rgba_frame, (width, canvas_height))
                    self.frame = rgba_frame
                else:
                    self.frame = np.zeros((canvas_height, width, 4), dtype=np.uint8)
                    self.width = width
            elif self.video_paused:
                self.frame = self.frame
            else:
                self.frame = np.zeros((canvas_height, width, 4), dtype=np.uint8)


    def set_slides(self, image_paths, duration=5):
        self.slides = image_paths
        self.slide_index = 0
        self.slide_duration = duration
        self.last_slide_time = time.time()


class Categorize(dspy.Signature):
    """Classify the intent of the command."""

    command: str = dspy.InputField()
    intent: Literal[
        "screen control (on/off)",
        "video content control",
        "powerpoint slide content control",
        "system rest/aeris go to sleep",  # black screen
        "experience wall control (on/off/content)",
        "content control on all screensvolume control",
    ] = dspy.OutputField()
    confidence: float = dspy.OutputField()


class CommandControl(dspy.Signature):
    """Determine the command details based on the input request and intention.

    Input:
    - request: actual request from the user
    Output:
    # Here are the valid responses:
    - play_video(video_name, screen_number)
    - pause_video(screen_number)
    - stop_video(screen_number)
    - control_presentation(presentation_name, screen_number)
    - experience_wall_control(action)  # valid actions: on/off
    - set_volume(volume_level)
    - system_sleep()

    [PRIORITY] Try your best to match file names and screen numbers. 
    [FALLBACK] If there's no matching file names, try using similar sounding phonetics.

    # Valid video files names:
    gca_video
    cw_aero_video
    pi_tech_video
    gat_video

    # Valid screen number are (1,2,3). However, please map it to (0, 1 and 2.)
    """
    request: str = dspy.InputField()

    command: str = dspy.OutputField(desc="Actual function and parameters to run")
    confidence: float = dspy.OutputField(
        desc="For downstream error handling or routing"
    )  # For downstream error handling or routing

# Initialize 3 panels
media_panels = [MediaPanel(i) for i in range(4)]
experience_wall = media_panels[-1]

# Background thread: single OpenCV window to display all panels
def display_loop():
    cv2.namedWindow("Media Dashboard", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Media Dashboard", canvas_width, canvas_height)

    # Function to overlay frame2 over frame1
    def overlay_frames(frame1, frame2):
        # Ensure both frames are the same size
        if frame1.shape != frame2.shape:
            raise ValueError("Frames must be the same size")

        # Create a copy of frame1 to modify
        combined_frame = frame1.copy()

        # Check where frame1 is transparent
        alpha1 = frame1[:, :, 3]  # Get the alpha channel of frame1
        mask = alpha1 == 0  # Create a mask where frame1 is transparent

        # Overlay frame2 where frame1 is transparent
        combined_frame[mask] = frame2[mask]  # Set pixels from frame2 where frame1 is transparent

        return combined_frame

    while True:
        # first, we fill it with the experience wall content
        experience_wall.update_frame(canvas_width)
        exp_frames = experience_wall.frame

        total_ratio = sum(width_ratios)
        panel_widths = [int(canvas_width * (r / total_ratio)) for r in width_ratios]
        frames = []
        for i, panel in enumerate(media_panels[:-1]):
            panel.update_frame(panel_widths[i])
            frames.append(panel.frame)
        panel_canvas = np.hstack(frames)

        # just to make sure the frames are same size
        panel_canvas = cv2.resize(panel_canvas, (canvas_width, canvas_height))
        exp_frames = cv2.resize(exp_frames, (canvas_width, canvas_height))
        canvas = overlay_frames(panel_canvas, exp_frames)

        cv2.imshow("Media Dashboard", canvas)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

threading.Thread(target=display_loop, daemon=True).start()

def log_function_call(func):
    def wrapper(*args, **kwargs):
        # Log the function name and arguments
        print(f"Calling function: {func.__name__}")
        print(f"Arguments: {args}, Keyword Arguments: {kwargs}")
        
        # Call the original function
        result = func(*args, **kwargs)
        
        # Optionally log the result
        print(f"Function {func.__name__} returned: {result}")
        
        return result
    return wrapper

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

@log_function_call
def play_video(video_name="", panel_index=0):
    config = configparser.ConfigParser()
    config.read("config.ini")

    video_dict = config["presentation"]
    if video_name:
        media_panels[panel_index].set_video(video_dict[video_name])

    if panel_index == 'all':
        for p in media_panels:
            p.set_video(video_dict[video_name])
            p.play()
    else:
        media_panels[panel_index].play()

def pause_video(panel_index=0):
    if panel_index == 'all':
        for p in media_panels:
            p.pause()
    else:
        media_panels[panel_index].pause()

def stop_video(panel_index=0):
    if panel_index == 'all':
        for p in media_panels:
            p.stop()
    else:
        media_panels[panel_index].stop()

def control_presentation():
    raise NotImplementedError

def experience_wall_control(action_name):
    raise NotImplementedError

def set_volume():
    raise NotImplementedError

def system_sleep():
    for p in media_panels:
        p.stop()


class VoiceAssistant:
    def __init__(self, log_to_memory, command_registry):

        self.log_to_memory = log_to_memory
        self.config = configparser.ConfigParser()
        self.config.read("config.ini")

        # DSPY related
        ollama_server_url = self.config["llm"]["endpoint_test"]
        self.lm = dspy.LM(
            self.config["llm"]["model"], 
            api_base=ollama_server_url, 
            api_key=""
        )
        dspy.configure(lm=self.lm)
        self.classify = dspy.Predict(Categorize)
        self.get_command_actions = dspy.ChainOfThought(CommandControl)

        # voice engine
        self.voice_engine = pyttsx3.init("sapi5")
        self.voices = self.voice_engine.getProperty("voices")
        self.voice_engine.setProperty('voice', self.voices[1].id)   #changing index, changes voices. 1 for female

        # dictionary of video files
        self.video_dict = self.config["presentation"]
        self.command_registry = command_registry

        self.recognizer = sr.Recognizer()
        self.mic_index = int(self.config["voice_control"]["mic_index"])

    def speak(self, audio):
        self.voice_engine.say(audio)
        self.voice_engine.runAndWait()

    def execute_command(self, command_str: str):
        try:
            # Safely parse the command string
            tree = ast.parse(command_str, mode="eval")
            if not isinstance(tree.body, ast.Call):
                raise ValueError("Command must be a function call")

            func_name = tree.body.func.id

            args = []
            for a in tree.body.args:
                args.append(a.value)

            kargs = {}
            for kw in tree.body.keywords:
                kargs[kw.arg] = ast.literal_eval(kw.value)

            if func_name not in self.command_registry:
                raise ValueError(f"Unknown command: {func_name}")

            self.command_registry[func_name](*args, **kargs)

        except Exception as e:
            print(f"Failed to execute command: {e}")

    def verify_user(self,):
        return True
        audio = self.recognizer.listen(source)
        transcript = self.recognizer.recognize_faster_whisper(audio)
        self.speak("Hello, identification please")
        raise NotImplementedError

    def unauthorized_access(self):
        self.speak("False identification, I do not recognize you.")


    def listen_for_wake_phrase(self, source):
        self.log_to_memory(f"Listening for wake phrases")
        wake_phrases = [
            "hello aeries",
            "hello aries",
            "hello iris",
            "hello aeris",
            "hello i miss",
            "hello irene",
            "hello ivy",
            "hello i mean",
            "hello im reese",
            "hello ill reach",
            "hello",  # use for testing
        ]
        while True:
            try:
                audio = self.recognizer.listen(source)
                transcript = self.recognizer.recognize_faster_whisper(audio)
                # transcript = self.recognizer.recognize_google(audio)
                if self.is_valid_command(transcript):
                    self.log_to_memory(f"You said: {transcript}")
                transcript = remove_punctuation(transcript)

                if any(w.lower() in transcript.lower() for w in wake_phrases):
                    self.log_to_memory("Wake word detected. Listening for command...")
                    validated = self.verify_user(source)
                    if validated:
                        self.try_to_recognize(source)
                    else:
                        self.unauthorized_access()
            except sr.UnknownValueError:
                self.log_to_memory("Didn't catch any sound.", level="INFO")
            except sr.RequestError as e:
                self.log_to_memory(f"Recognition error: {e}", level="ERROR")
                self.speak("Recognition error.")
            except Exception as e:
                self.log_to_memory(f"Unhandled error: {e}", level="ERROR")
                raise e

    def is_valid_command(self, message):
        """Check if the log message is valid."""
        return message and message.strip() not in ['', '...', '... ... ... ...']

    def try_to_recognize(self, source, repeated=False):


        self.recognizer.adjust_for_ambient_noise(source, duration=1)
        self.recognizer.pause_threshold = 2  # add some extra time to allow brain to process more
        if not repeated:
            self.speak('Hello, you have called for me. How can i help you?')
        
        self.log_to_memory("You may speak your command now.")
        audio = self.recognizer.listen(source, )
        command = self.recognizer.recognize_faster_whisper(audio).lower()
        self.log_to_memory(f"Command received: {command}")

        if self.is_valid_command(command):
            command_res = self.get_command_actions(
                request=command,
            )
            if command_res.confidence <= float(self.config["voice_control"]["confidence"]):
                self.speak("I'm sorry, could you repeat please?")
                self.try_to_recognize(source, repeated=True)

            self.speak(f"Okay running: {command}")
            self.log_to_memory(f"Generated command: {command_res}")
            # if it passes the conf threshold
            self.execute_command(command_res.command)

        self.recognizer.pause_threshold = 0.8  # reset to default

    def run(self):
        try:
            with sr.Microphone(device_index=self.mic_index) as source:
                self.log_to_memory("Calibrating for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
                self.log_to_memory("Always listening for wake word...")
                self.listen_for_wake_phrase(source)
        except Exception as mic_error:
            self.log_to_memory(f"Microphone initialization failed: {mic_error}", level="ERROR")

def run_aeris():

    log_to_memory("")
    log_to_memory(" ---------------- ")
    log_to_memory(" STARTING AERIS SERVICE")
    log_to_memory(" ---------------- ")
    log_to_memory("")

    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"Microphone {index}: {name}")

    # if mic_index is None:
    #     mic_index = input("Which mic would you like to use? :") or 8
    #     mic_index = int(mic_index)
    #
    command_registry: dict[str, Callable] = {
        "play_video": play_video,
        "pause_video": pause_video,
        "stop_video": stop_video,
        "control_presentation": control_presentation,
        "experience_wall_control": experience_wall_control,
        "set_volume": set_volume,
        "system_sleep": system_sleep,
    }

    aeris = VoiceAssistant(
        log_to_memory=log_to_memory, 
        command_registry=command_registry
    )
    aeris.run()



# if there's powerpoint slide
def convert_pptx_to_images(pptx_path: str) -> list[str]:
    output_dir = Path(tempfile.mkdtemp())
    ppt = comtypes.client.CreateObject("PowerPoint.Application")
    ppt.Visible = 1
    presentation = ppt.Presentations.Open(pptx_path, WithWindow=False)
    
    export_path = str(output_dir / "slide")
    presentation.SaveAs(export_path, 17)  # 17 = ppSaveAsJPG
    presentation.Close()
    ppt.Quit()

    # Collect all slide image paths
    slide_images = sorted(output_dir.glob("*.JPG"))
    return [str(p) for p in slide_images]

def convert_pptx_to_images_pure(pptx_path, width=1280, height=720) -> list[str]:
    prs = Presentation(pptx_path)
    output_dir = Path(tempfile.mkdtemp())

    slide_images = []
    for idx, slide in enumerate(prs.slides):
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)

        for shape in slide.shapes:
            if shape.shape_type == MSO_SHAPE_TYPE.TEXT_BOX or shape.has_text_frame:
                if shape.text_frame:
                    text = shape.text_frame.text
                    left = int(shape.left * width / prs.slide_width)
                    top = int(shape.top * height / prs.slide_height)
                    font_size = 24
                    draw.text((left, top), text, fill="black")  # Add font if needed

            elif shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                img_stream = shape.image.blob
                with Image.open(io.BytesIO(img_stream)) as shape_img:
                    shape_img = shape_img.convert("RGB")
                    shape_img = shape_img.resize((shape.width // 9525, shape.height // 9525))
                    img.paste(shape_img, (shape.left // 9525, shape.top // 9525))

        image_path = output_dir / f"slide_{idx+1}.jpg"
        img.save(image_path)
        slide_images.append(str(image_path))

    return slide_images

@app.route("/")
def index():
    return render_template("dashboard.html", panels=media_panels)


@app.route("/panel/<int:panel_id>/browse", methods=["POST"])
def browse_file(panel_id):
    print(f"Browse clicked for panel {panel_id}")
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(
        filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov"), ("All files", "*.*")]
    )
    root.destroy()
    print(f"Selected file: {file_path}")
    if file_path:
        media_panels[panel_id].set_video(file_path)
    return redirect(url_for("index"))

@app.route("/panel/<int:panel_id>/<action>", methods=["POST"])
def control_panel(panel_id, action):
    panel = media_panels[panel_id]
    if action == "play":
        panel.play()
    elif action == "pause":
        panel.pause()
    elif action == "stop":
        panel.stop()
    elif action == "clear_slides":
        panel.clear_slides()

    return redirect(url_for("index"))


@app.route("/panel/<int:panel_id>/upload_pptx", methods=["POST"])
def upload_pptx(panel_id):
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_path = filedialog.askopenfilename(filetypes=[("PowerPoint files", "*.pptx")])
    root.destroy()

    if file_path:
        slide_images = convert_pptx_to_images_pure(file_path)
        media_panels[panel_id].set_slides(slide_images, duration=5)  # set default time here

    return redirect(url_for("index"))

@app.route("/logs")
def view_logs():
    num_lines = 15
    try:
        with log_lock:
            # Get the last num_lines from memory
            lines = log_storage
            # Ensure all lines are strings
            lines = [line for line in lines if isinstance(line, str)]
    except Exception as e:
        lines = [f"Error reading log: {e}"]

    # Join the lines safely
    return "<br>".join(line.strip() for line in lines)

if __name__ == "__main__":
    print(EXEC_PLATFORM)

    assistant_thread = threading.Thread(
        target=run_aeris, 
        daemon=True,
    )
    assistant_thread.start()

    app.run(debug=True, use_reloader=True)
