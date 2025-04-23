
import cv2
import numpy as np

# Set paths to 3 video files
video_paths = [
    "video1.mp4",
    "video2.mp4",
    "video3.mp4"
]

# Load video capture objects
caps = [cv2.VideoCapture(path) for path in video_paths]

# Check they opened
for i, cap in enumerate(caps):
    if not cap.isOpened():
        print(f"Error opening video {i+1}")
        exit()

# Get screen dimensions (adjust as needed)
screen_width = 1920 # 3024
screen_height = 1080 # 672
zone_width = screen_width // 3

# Create fullscreen OpenCV window
cv2.namedWindow("SplitScreen", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("SplitScreen", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    frames = []
    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()

        # Resize each frame to fit one zone
        resized = cv2.resize(frame, (zone_width, screen_height))
        frames.append(resized)

    # Concatenate horizontally to form the split screen
    combined = np.hstack(frames)

    # Show on screen
    cv2.imshow("SplitScreen", combined)

    # Wait for key or delay for frame sync
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Release resources
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
