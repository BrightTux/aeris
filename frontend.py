
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QFileDialog
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QUrl, Qt

class MediaPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()

        # Buttons
        self.load_btn = QPushButton("Load File")
        self.play_btn = QPushButton("Play")
        self.pause_btn = QPushButton("Pause")
        self.stop_btn = QPushButton("Stop")

        self.load_btn.clicked.connect(self.load_file)
        self.play_btn.clicked.connect(self.play)
        self.pause_btn.clicked.connect(self.pause)
        self.stop_btn.clicked.connect(self.stop)

        # Display Area
        self.video_widget = QVideoWidget()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        # Media Player
        self.player = QMediaPlayer()
        self.player.setVideoOutput(self.video_widget)

        # Add widgets to layout
        self.layout.addWidget(self.load_btn)
        self.layout.addWidget(self.video_widget)
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.play_btn)
        self.layout.addWidget(self.pause_btn)
        self.layout.addWidget(self.stop_btn)

        self.setLayout(self.layout)

        # Hide one by default
        self.image_label.hide()

    def load_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Media File", "", "Media Files (*.mp4 *.avi *.jpg *.png)")
        if file:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                self.image_label.hide()
                self.video_widget.show()
                self.player.setMedia(QMediaContent(QUrl.fromLocalFile(file)))
                self.play()
            else:
                self.player.stop()
                self.video_widget.hide()
                self.image_label.show()
                pixmap = QPixmap(file).scaled(480, 270, Qt.KeepAspectRatio)
                self.image_label.setPixmap(pixmap)

    def play(self):
        self.player.play()

    def pause(self):
        self.player.pause()

    def stop(self):
        self.player.stop()

class MainApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Split Media Player")
        self.setGeometry(100, 100, 1440, 480)

        layout = QHBoxLayout()

        # Create 3 media panels side by side
        for _ in range(3):
            panel = MediaPanel()
            layout.addWidget(panel)

        self.setLayout(layout)

app = QApplication(sys.argv)
window = MainApp()
window.show()
sys.exit(app.exec_())
