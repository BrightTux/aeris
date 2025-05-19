import configparser
import cv2
import copy
import io
import logging
import numpy as np
import os
import platform
import random
import string
import subprocess
import tempfile
import threading
import time
import traceback
import simpleaudio as sa

from PIL import Image, ImageDraw
from collections import deque
from flask import Flask, render_template, redirect, url_for, request
from pathlib import Path
from screeninfo import get_monitors
from tkinter import Tk, filedialog
from typing import Callable
from pydub import AudioSegment

EXEC_PLATFORM = platform.system()
config = configparser.ConfigParser()
config.read("config.ini")
USE_GOOGLE = int(config["misc"]["use_google"])
FULLSCREEN = int(config["misc"]["fullscreen"])
DEBUG_TEST = False

if EXEC_PLATFORM == "Windows":
    import pygetwindow as gw

    # audio/volume control
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

app = Flask(__name__)

# Width ratios for each panel (default: equal)
width_ratios = [1, 1, 1, 1]  # Can be adjusted dynamically if needed
canvas_width = 3064
canvas_height = 672
total_ratio = sum(width_ratios)
panel_widths = [int(canvas_width * (r / total_ratio)) for r in width_ratios]
panel_widths.append(canvas_width)  # for the exp wall

# In-memory log storage
log_storage = deque(maxlen=20)
log_lock = threading.Lock()

# Configure logging
logging.basicConfig(level=logging.INFO)
voice_logger = logging.getLogger("voice_assistant")


def format_log_message(level, message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")  # Format the timestamp
    return f"[{timestamp}] [{level}] *{message}"


def log_to_memory(message, level="INFO"):
    formatted_message = format_log_message(level, message)
    with log_lock:
        try:
            previous_msg = log_storage[-1].split("*")[-1]
            if previous_msg == message:
                return
        except IndexError:
            pass  # when launched, the deque is empty
        log_storage.append(formatted_message)


def write_logs_to_file():
    log_file = "voice_assistant.log"
    while True:
        with log_lock:
            with open(log_file, "a", encoding="utf-8", errors="ignore") as f:
                for line in log_storage:
                    f.write(line + "\n")
        time.sleep(60)  # Write logs to file every 60 seconds


log_thread = threading.Thread(target=write_logs_to_file, daemon=True)
log_thread.start()


class MediaPanel:
    def __init__(self, panel_id, width, exp_wall: bool = False):
        self.panel_id = panel_id
        self.filepath = ""
        self.cap = None
        self.playing = False
        self.video_paused = False
        self.frame = np.zeros((canvas_height, width, 3), dtype=np.uint8)
        self.width = width
        self.lock = threading.Lock()
        self.exp_wall = exp_wall
        if self.exp_wall:
            self.audio_cap = None
            self.audio_play_obj = None

        self.slides = []
        self.slide_index = 0
        self.slide_duration = 5
        self.fade_duration = 1
        self.last_slide_time = time.time()

    def set_video(self, path):
        with self.lock:
            self.filepath = path
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(path)
            self.playing = False
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.wait_time = 1.0 / self.fps  # count time in seconds
            self.last_read_time = 0
            if self.exp_wall:
                self.audio_cap = AudioSegment.from_file(path)

    def play(self):
        with self.lock:
            self.playing = True
            self.video_paused = False
            try:
                self.last_video_time_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            except Exception:
                self.last_video_time_ms = 0
            self.play_audio_from(int(self.last_video_time_ms))

    def pause(self):
        with self.lock:
            self.playing = False
            self.video_paused = True
            if self.exp_wall:
                if self.audio_play_obj:
                    self.audio_play_obj.stop()

    def stop(self):
        with self.lock:
            self.playing = False
            self.video_paused = False
            self.frame = None
            if self.cap:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if self.exp_wall:
                if self.audio_play_obj:
                    self.audio_play_obj.stop()

    def clear_slides(self):
        with self.lock:
            self.slides = []

    def play_audio_from(self, ms):
        segment = self.audio_cap[ms:]
        self.audio_play_obj = sa.play_buffer(
            segment.raw_data,
            num_channels=segment.channels,
            bytes_per_sample=segment.sample_width,
            sample_rate=segment.frame_rate,
        )

    def update_frame(
        self,
        start_time,
    ):
        with self.lock:
            if self.slides:
                now = time.time()
                if now - self.last_slide_time > self.slide_duration:
                    # Update slide index and last slide time
                    self.last_slide_time = now
                    next_slide_index = (self.slide_index + 1) % len(self.slides)

                    # Load current and next images
                    current_img = cv2.imread(self.slides[self.slide_index])
                    # next_img = cv2.imread(self.slides[next_slide_index])

                    # Convert images to BGRA
                    # current_rgba = cv2.cvtColor(current_img, cv2.COLOR_BGR2BGRA)
                    # next_rgba = cv2.cvtColor(next_img, cv2.COLOR_BGR2BGRA)

                    # Resize images to fit the canvas
                    current_rgba = cv2.resize(current_img, (self.width, canvas_height))
                    # next_rgba = cv2.resize(next_rgba, (self.width, canvas_height))

                    # Fade out current image and fade in next image
                    # for alpha in np.linspace(
                    #    1, 0, int(self.fade_duration * 30)
                    # ):  # Assuming 30 FPS
                    #    # Create a blended image
                    #    blended = cv2.addWeighted(
                    #        current_rgba, alpha, next_rgba, 1 - alpha, 0
                    #    )
                    #    self.frame = blended
                    self.frame = current_rgba

                    # Update slide index after the transition
                    self.slide_index = next_slide_index

                else:
                    # Display the current slide
                    img = cv2.imread(self.slides[self.slide_index])
                    # rgba_image = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                    self.frame = cv2.resize(img, (self.width, canvas_height))

            elif self.playing and self.cap and self.cap.isOpened():
                current_time = time.time() - start_time
                if current_time - self.last_read_time >= self.wait_time:
                    ret, frame = self.cap.read()
                    if not ret:
                        # if finish playing, loop it
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self.cap.read()
                        self.last_read_time = current_time
                        self.last_video_time_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                    if ret:
                        if (
                            frame.shape[1] != self.width
                            or frame.shape[0] != canvas_height
                        ):
                            frame = cv2.resize(frame, (self.width, canvas_height))
                        self.frame = frame
                        self.last_read_time = current_time
                        self.last_video_time_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                    else:
                        self.frame = None
                        self.last_read_time = current_time
                        self.last_video_time_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)

            elif self.video_paused:
                self.frame = self.frame
                if self.exp_wall:
                    self.audio_play_obj.stop()

            else:
                self.frame = None

    def set_slides(self, image_paths, duration=5):
        self.slides = image_paths
        self.slide_index = 0
        self.slide_duration = duration
        self.last_slide_time = time.time()


# Initialize 4 panels + exp wall panel
media_panels = [MediaPanel(panel_id=i, width=panel_widths[i]) for i in range(5)]
media_panels[-1].exp_wall = True
experience_wall = media_panels[-1]


# Background thread: single OpenCV window to display all panels
def display_loop():
    cv2.namedWindow("Media Dashboard", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Media Dashboard", canvas_width, canvas_height)

    # Move display window to the target monitor
    # Function to get the position of the monitor with the specified name
    def get_monitor_position(monitor_name):
        for monitor in get_monitors():
            print(f"DEBUG: {monitor=}, {monitor_name=}, {monitor.name == monitor_name}")
            if monitor_name == monitor.name:
                return (monitor.x, monitor.y, monitor.width, monitor.height)
        return None

    # Get the position of the HDMI-1 monitor
    config = configparser.ConfigParser()
    config.read("config.ini")

    print(config["screen"]["target"])
    target_monitor = get_monitor_position(config["screen"]["target"])

    if target_monitor is None:
        print("target monitor not found.")
    else:
        # Get the window with the title "abc"
        try:
            window = gw.getWindowsWithTitle("Media Dashboard")[
                0
            ]  # Get the first window with the title

            # Move the window to the HDMI-1 monitor
            window.moveTo(target_monitor[0], target_monitor[1])
            window.maximize()

            print(f"Moved window '{window.title}' to HDMI-1 monitor.")
        except IndexError:
            print("Window with title 'Media Dashboard' not found.")

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
        combined_frame[mask] = frame2[
            mask
        ]  # Set pixels from frame2 where frame1 is transparent

        return combined_frame

    start_time = time.time()
    while True:
        # first, we fill it with the experience wall content
        experience_wall.update_frame(start_time)
        exp_frames = experience_wall.frame
        if exp_frames is None:
            exp_frames = np.zeros(
                (canvas_height, experience_wall.width, 3), dtype=np.uint8
            )
        backup_frame = copy.deepcopy(exp_frames)

        # just to make sure the frames are same size
        if exp_frames.shape[1] != canvas_width or exp_frames.shape[0] != canvas_height:
            exp_frames = cv2.resize(exp_frames, (canvas_width, canvas_height))

        for i, panel in enumerate(media_panels[:-1]):
            x_start = i * panel.width
            x_end = x_start + panel.width

            panel.update_frame(start_time)
            if panel.frame is not None:
                exp_frames[:, x_start:x_end] = panel.frame

        cv2.imshow("Media Dashboard", exp_frames)
        if FULLSCREEN:
            cv2.setWindowProperty(
                "Media Dashboard", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
        # Check for 'Ctrl + Q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
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


@log_function_call
def play_video(video_name="", panel_index=0, *args, **kwargs):
    config = configparser.ConfigParser()
    config.read("config.ini")

    video_dict = config["presentation"]
    if video_name:
        media_panels[panel_index].set_video(video_dict[video_name])

    if panel_index == "all":
        for p in media_panels:
            p.set_video(video_dict[video_name])
            p.play()
    else:
        media_panels[panel_index].play()


def pause_video(panel_index=0, *args, **kwargs):
    if panel_index == "all":
        for p in media_panels:
            p.pause()
    else:
        media_panels[panel_index].pause()


def stop_video(panel_index=0, *args, **kwargs):
    if panel_index == "all":
        for p in media_panels:
            p.stop()
    else:
        media_panels[panel_index].stop()
        if panel_index == -1:
            return
        exp_wall_paused = media_panels[-1].video_paused
        if exp_wall_paused:
            print("was it paused?")
            media_panels[-1].play()
            time.sleep(0.1)
            media_panels[-1].pause()


def experience_wall_control(action, *args, **kwargs):
    if action.lower() == "on":
        play_video(video_name="experience_wall", panel_index=-1)
    elif action.lower() == "off":
        stop_video(panel_index=-1)


def set_volume(volume_level, *args, **kwargs):
    # Ensure volume_level is between 0.0 and 1.0
    if not (0.0 <= volume_level <= 1.0):
        raise ValueError("Volume level must be between 0.0 and 1.0")

    # Get the audio devices
    speaker_devices = AudioUtilities.GetSpeakers()

    # Get the volume interface for the default audio device
    speaker_interface = speaker_devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None
    )

    # Cast to IAudioEndpointVolume to manipulate the volume
    volume = cast(speaker_interface, POINTER(IAudioEndpointVolume))

    # Set the master volume level
    volume.SetMasterVolumeLevelScalar(volume_level, None)


def system_sleep(*args, **kwargs):
    for p in media_panels:
        p.stop()


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
                    draw.text((left, top), text, fill="black")  # Add font if needed

            elif shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                img_stream = shape.image.blob
                with Image.open(io.BytesIO(img_stream)) as shape_img:
                    shape_img = shape_img.convert("RGB")
                    shape_img = shape_img.resize(
                        (shape.width // 9525, shape.height // 9525)
                    )
                    img.paste(shape_img, (shape.left // 9525, shape.top // 9525))

        image_path = output_dir / f"slide_{idx + 1}.jpg"
        img.save(image_path)
        slide_images.append(str(image_path))

    return slide_images


@app.route("/")
def index():
    config = configparser.ConfigParser()
    return render_template("dashboard.html", panels=media_panels, config=config)


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
        panel.video_paused = False
    elif action == "pause":
        panel.pause()
    elif action == "stop":
        panel.stop()
        if panel == media_panels[-1]:
            return redirect(url_for("index"))
        print("debug:", media_panels[-1].video_paused)
        exp_wall_paused = media_panels[-1].video_paused
        if exp_wall_paused:
            print("was it paused?")
            media_panels[-1].play()
            time.sleep(0.1)
            media_panels[-1].pause()
    elif action == "clear_slides":
        panel.clear_slides()

    return redirect(url_for("index"))


@app.route("/bulk_action", methods=["POST"])
def bulk_action():
    action = request.form.get("action")
    panel_ids = request.form.getlist("panel_ids")  # this is a list of strings

    for panel_id_str in panel_ids:
        try:
            panel_id = int(panel_id_str)
            panel = media_panels[panel_id]
            if action == "play":
                panel.play()
            elif action == "pause":
                panel.pause()
            elif action == "stop":
                panel.stop()
            elif action == "clear_slides":
                panel.clear_slides()
        except (ValueError, IndexError):
            # Handle invalid panel ID
            continue

    return redirect(url_for("index"))


@app.route("/panel/<int:panel_id>/upload_pptx", methods=["POST"])
def upload_pptx(panel_id):
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_path = filedialog.askopenfilename(filetypes=[("PowerPoint files", "*.pptx")])
    root.destroy()

    if file_path:
        # slide_images = convert_pptx_to_images_pure(file_path)
        slide_images = convert_pptx_to_images(file_path)
        media_panels[panel_id].set_slides(
            slide_images, duration=5
        )  # set default time here

    return redirect(url_for("index"))


@app.route("/panel/<int:panel_id>/upload_images", methods=["POST"])
def upload_images(panel_id):
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    folder_path = filedialog.askdirectory()  # Ask for a directory instead of a file
    root.destroy()

    if folder_path:
        # Get a list of image files in the selected directory
        image_extensions = (
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".bmp",
            ".webp",
        )  # Add more extensions if needed
        slide_images = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(image_extensions)
        ]

        media_panels[panel_id].set_slides(
            slide_images, duration=5
        )  # Set default time here

    return redirect(url_for("index"))


@app.route("/logs")
def view_logs():
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
    app.run(debug=False, use_reloader=False)
