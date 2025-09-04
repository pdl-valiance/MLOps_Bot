# ui/eda/base_eda_page.py

import pandas as pd
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton,
                             QComboBox, QHBoxLayout, QLabel, QScrollArea)
from app_state import AppState

class BaseEdaPage(QWidget):
    def __init__(self, app_state: AppState, title: str):
        super().__init__()
        self.app_state = app_state
        self.df = None
        self.grouping_cols = []
        self.date_col = None
        self.numeric_cols = []

        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(15)

        # --- Controls ---
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Select Segment:"))
        self.group_combo = QComboBox()
        controls_layout.addWidget(self.group_combo, 1)
        
        self.run_button = QPushButton(f"Generate {title}")
        self.run_button.clicked.connect(self.run_analysis)
        controls_layout.addWidget(self.run_button)
        page_layout.addLayout(controls_layout)

        # --- Plotting Area ---
        self.plot_area = QScrollArea()
        self.plot_area.setWidgetResizable(True)
        self.plot_area.setStyleSheet("QScrollArea { border: none; }")
        plot_widget = QWidget()
        self.plot_layout = QVBoxLayout(plot_widget)
        self.plot_layout.setAlignment(Qt.AlignTop)
        self.plot_area.setWidget(plot_widget)
        page_layout.addWidget(self.plot_area)

    def set_data(self, df: pd.DataFrame):
        self.df = df
        if df is None:
            self.group_combo.clear()
            return
        
        # Find grouping columns, date, and numeric columns from the aggregated data
        self.grouping_cols = [col for col in df.columns if df[col].dtype == 'object']
        self.date_col = next((col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])), None)
        self.numeric_cols = df.select_dtypes(include=pd.np.number).columns.tolist()

        self.group_combo.clear()
        self.group_combo.addItem("All Segments") # Add the 'All' option
        if self.grouping_cols:
            unique_segments = df[self.grouping_cols].drop_duplicates()
            for index, row in unique_segments.iterrows():
                self.group_combo.addItem(" | ".join(row.astype(str)))
        else:
            self.group_combo.addItem("Overall") # If no groups, this is the only option

    def run_analysis(self):
        """ This method should be implemented by subclasses. """
        raise NotImplementedError("This method must be implemented by a subclass.")

    def clear_plots(self):
        """ Helper to clear the plot area before generating new ones. """
        while self.plot_layout.count():
            child = self.plot_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
