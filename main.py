import sys
import cv2
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QListWidget,
    QFileDialog, QVBoxLayout, QWidget
)
from PySide6.QtGui import QIcon


class FaceDetector:
    def __init__(self):
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades +
                                              "haarcascade_frontalface_default.xml")

    def has_face(self, image_path: str) -> bool:
        img = cv2.imread(image_path)
        if img is None:
            return False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        return len(faces) > 0


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("얼굴 탐지기 (Face Detector)")
        self.setMinimumSize(500, 400)

        self.detector = FaceDetector()

        self.button = QPushButton("이미지 선택 (여러 개 가능)")
        self.button.clicked.connect(self.open_images)

        self.result_list = QListWidget()

        layout = QVBoxLayout()
        layout.addWidget(self.button)
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
            has_face = self.detector.has_face(path)
            status = "✅ 얼굴 있음" if has_face else "❌ 없음"
            self.result_list.addItem(f"{path} → {status}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
