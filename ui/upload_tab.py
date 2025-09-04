# ui/upload_tab.py
import os
import pandas as pd
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QFrame, QLabel, QFileDialog)
from PyQt5.QtCore import Qt
from app_state import AppState

class UploadDataTab(QWidget):
    def __init__(self, app_state: AppState):
        super().__init__()
        self.app_state = app_state
        self.df = None
        self.setAcceptDrops(True)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)

        title = QLabel("Upload Your Dataset")
        title.setObjectName("h1_label")
        layout.addWidget(title, alignment=Qt.AlignCenter)
        
        self.drop_area = QFrame()
        self.drop_area.setObjectName("card")
        self.drop_area.setMinimumSize(450, 250)
        self.drop_area.setStyleSheet("border: 2px dashed #CED4DA;")
        drop_layout = QVBoxLayout(self.drop_area)
        drop_layout.setAlignment(Qt.AlignCenter)
        drop_layout.setSpacing(15)

        drop_label = QLabel("Drag & Drop Your CSV File Here")
        drop_label.setStyleSheet("font-size: 18px; color: #868E96;")
        drop_layout.addWidget(drop_label, alignment=Qt.AlignCenter)
        
        or_label = QLabel("Or")
        or_label.setStyleSheet("font-size: 14px; color: #ADB5BD;")
        drop_layout.addWidget(or_label, alignment=Qt.AlignCenter)

        self.browse_button = QPushButton("Browse Files")
        self.browse_button.setMinimumWidth(150)
        self.browse_button.clicked.connect(self.open_file_dialog)
        drop_layout.addWidget(self.browse_button, alignment=Qt.AlignCenter)

        layout.addWidget(self.drop_area)

        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.setEnabled(False)
        self.analyze_button.clicked.connect(self.start_analysis)
        layout.addWidget(self.analyze_button, alignment=Qt.AlignCenter)

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV Files (*.csv)")
        if file_path: self.process_file(file_path)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.acceptProposedAction()
        else: event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files and files[0].endswith('.csv'): self.process_file(files[0])

    def process_file(self, file_path):
        try:
            self.df = pd.read_csv(file_path, low_memory=False)
            self.analyze_button.setEnabled(True)
            base_name = os.path.basename(file_path)
            self.drop_area.setStyleSheet("border: 2px solid #4C6EF5;")
            self.browse_button.setText(f"Loaded: {base_name[:30]}")
        except Exception as e:
            self.analyze_button.setEnabled(False)
            print(f"Error: {e}")

    def start_analysis(self):
        if self.df is not None:
            # CORRECTED: Called the correct method name from app_state.py
            self.app_state.load_new_data(self.df)