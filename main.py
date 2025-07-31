import sys
import cv2
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QListWidget,
    QFileDialog, QVBoxLayout, QLabel, QWidget, QScrollArea
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt


class FaceDetector:
    def __init__(self):
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades +
                                              "haarcascade_frontalface_default.xml")

    def detect_and_draw_faces(self, image_path: str):
        img = cv2.imread(image_path)
        if img is None:
            return None, False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        return img, len(faces) > 0


def cvimg_to_qpixmap(cv_img):
    """Convert BGR OpenCV image to QPixmap"""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("얼굴 박스 표시기 (Face Box Viewer)")
        self.setMinimumSize(800, 600)

        self.detector = FaceDetector()

        self.button = QPushButton("이미지 선택")
        self.button.clicked.connect(self.open_images)

        self.result_list = QListWidget()
        self.image_label = QLabel(alignment=Qt.AlignCenter)
        self.image_label.setMinimumHeight(300)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.image_label)

        layout = QVBoxLayout()
        layout.addWidget(self.button)
        layout.addWidget(scroll)
        layout.addWidget(self.result_list)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def open_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "이미지 선택",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not paths:
            return

        self.result_list.clear()

        for path in paths:
            img_with_boxes, has_face = self.detector.detect_and_draw_faces(path)
            status = "✅ 얼굴 있음" if has_face else "❌ 없음"
            self.result_list.addItem(f"{path} → {status}")

            if img_with_boxes is not None:
                pixmap = cvimg_to_qpixmap(img_with_boxes).scaledToWidth(
                    700, Qt.SmoothTransformation)
                self.image_label.setPixmap(pixmap)
