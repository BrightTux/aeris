import configparser
import cv2
import logging
import numpy as np
import os
import platform
import string
import threading
import time

from collections import deque
from flask import Flask, render_template, redirect, url_for, request
from screeninfo import get_monitors
from tkinter import Tk, filedialog

EXEC_PLATFORM = platform.system()
config = configparser.ConfigParser()
config.read("config.ini")
USE_GOOGLE = int(config["misc"]["use_google"])
FULLSCREEN = int(config["misc"]["fullscreen"])


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
    def __init__(self, panel_id, width):
        self.panel_id = panel_id
        self.filepath = ""
        self.cap = None
        self.playing = False
        self.video_paused = False
        self.frame = np.zeros((canvas_height, width, 4), dtype=np.uint8)
        self.width = width
        self.lock = threading.Lock()

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
            self.wait_time = int(1 / self.fps)  # count time in seconds
            self.last_read_time = 0

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
                    next_img = cv2.imread(self.slides[next_slide_index])

                    # Convert images to BGRA
                    current_rgba = cv2.cvtColor(current_img, cv2.COLOR_BGR2BGRA)
                    next_rgba = cv2.cvtColor(next_img, cv2.COLOR_BGR2BGRA)

                    # Resize images to fit the canvas
                    current_rgba = cv2.resize(current_rgba, (self.width, canvas_height))
                    next_rgba = cv2.resize(next_rgba, (self.width, canvas_height))

                    # Fade out current image and fade in next image
                    for alpha in np.linspace(
                        1, 0, int(self.fade_duration * 30)
                    ):  # Assuming 30 FPS
                        # Create a blended image
                        blended = cv2.addWeighted(
                            current_rgba, alpha, next_rgba, 1 - alpha, 0
                        )
                        self.frame = blended

                    # Update slide index after the transition
                    self.slide_index = next_slide_index

                else:
                    # Display the current slide
                    img = cv2.imread(self.slides[self.slide_index])
                    rgba_image = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                    self.frame = cv2.resize(rgba_image, (self.width, canvas_height))

            elif self.playing and self.cap and self.cap.isOpened():
                current_time = time.time() - start_time
                if current_time - self.last_read_time >= self.wait_time:
                    ret, frame = self.cap.read()
                    if not ret:
                        # if finish playing, loop it
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self.cap.read()
                    if ret:
                        rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                        rgba_frame = cv2.resize(rgba_frame, (self.width, canvas_height))
                        self.frame = rgba_frame
                    else:
                        self.frame = np.zeros(
                            (canvas_height, self.width, 4), dtype=np.uint8
                        )
                self.last_read_time = current_time
            elif self.video_paused:
                self.frame = self.frame
            else:
                self.frame = np.zeros((canvas_height, self.width, 4), dtype=np.uint8)

    def set_slides(self, image_paths, duration=5):
        self.slides = image_paths
        self.slide_index = 0
        self.slide_duration = duration
        self.last_slide_time = time.time()


# Initialize 4 panels + exp wall panel
media_panels = [MediaPanel(panel_id=i, width=panel_widths[i]) for i in range(5)]
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

        frames = []
        for i, panel in enumerate(media_panels[:-1]):
            panel.update_frame(start_time)
            frames.append(panel.frame)
        panel_canvas = np.hstack(frames)

        # just to make sure the frames are same size
        panel_canvas = cv2.resize(panel_canvas, (canvas_width, canvas_height))
        exp_frames = cv2.resize(exp_frames, (canvas_width, canvas_height))
        canvas = overlay_frames(panel_canvas, exp_frames)

        cv2.imshow("Media Dashboard", canvas)
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


def remove_punctuation(text):
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


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
    elif action == "pause":
        panel.pause()
    elif action == "stop":
        panel.stop()
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
