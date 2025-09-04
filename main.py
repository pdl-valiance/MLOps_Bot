# main.py

import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QFrame, QLabel, QStackedWidget,
                             QToolButton, QMenu, QAction)
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QPropertyAnimation, QEasingCurve

# --- Hot-Reloading Imports ---
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- App Components ---
from app_state import AppState
from ui.upload_tab import UploadDataTab
from ui.overview_tab import DataOverviewTab
from ui.plot_tab import PlotTab # NEW IMPORT for PlotTab
from ui.column_selection_tab import ColumnSelectionTab
from ui.rename_tab import RenameTab
from ui.data_cleaner_tab import DataCleanerTab
from ui.aggregation_tab import AggregationTab
from ui.exploratory_analysis_tab import (TimeSeriesPage, DemandPatternsPage, DataQualityPage,
                                         ProductInsightsPage, FeatureInsightsPage, VolumeAnalysisPage,
                                         AutoMLPage, ParetoAnalysisPage)

# --- NEW: Import the AI Analyst Tab ---
from ui.ai_analyst_tab import AIAnalystTab

# --- Handler for Stylesheet Hot-Reloading ---
class StyleReloader(QObject):
    update_style_signal = pyqtSignal(str)

class QssEventHandler(FileSystemEventHandler):
    def __init__(self, reloader):
        super().__init__()
        self.reloader = reloader

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('main.qss'):
            print("Stylesheet changed! Reloading...")
            try:
                with open(event.src_path, 'r') as f:
                    stylesheet = f.read()
                    self.reloader.update_style_signal.emit(stylesheet)
            except Exception as e:
                print(f"Error reloading stylesheet: {e}")

# --- Main Application Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Valiance EDA")
        self.setGeometry(100, 100, 1600, 950)

        self.SIDEBAR_EXPANDED_WIDTH = 250
        self.SIDEBAR_COLLAPSED_WIDTH = 60 # Increased for better icon visibility
        self.sidebar_is_expanded = True

        self.app_state = AppState()
        self.load_stylesheet()

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.main_layout = QHBoxLayout(main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.plot_tab_instance = PlotTab(self.app_state)

        # --- UPDATED: Pages list with new AI Analyst Tab at the end ---
        self.pages = [
            UploadDataTab(self.app_state),          # 0
            DataOverviewTab(),                      # 1
            ColumnSelectionTab(self.app_state),     # 2
            RenameTab(self.app_state),              # 3
            DataCleanerTab(self.app_state),         # 4
            AggregationTab(self.app_state),         # 5
            self.plot_tab_instance,                 # 6
            TimeSeriesPage(self.app_state),         # 7
            DemandPatternsPage(self.app_state),     # 8
            DataQualityPage(self.app_state),        # 9
            ProductInsightsPage(self.app_state),    # 10
            FeatureInsightsPage(self.app_state),    # 11
            VolumeAnalysisPage(self.app_state),     # 12
            AutoMLPage(self.app_state),             # 13
            ParetoAnalysisPage(self.app_state),     # 14
            AIAnalystTab(self.app_state)            # 15 (NEW)
        ]

        self.sidebar = self.create_sidebar()
        self.main_layout.addWidget(self.sidebar)

        self.content_area = QStackedWidget()
        for page in self.pages:
            self.content_area.addWidget(page)

        self.main_layout.addWidget(self.content_area)

        self.connect_signals()
        self.setup_hot_reloader()

    def create_sidebar(self):
        sidebar = QFrame(); sidebar.setObjectName("sidebar")
        sidebar.setMinimumWidth(self.SIDEBAR_EXPANDED_WIDTH)
        sidebar.setMaximumWidth(self.SIDEBAR_EXPANDED_WIDTH)

        sidebar_layout = QVBoxLayout(sidebar); sidebar_layout.setAlignment(Qt.AlignTop)
        sidebar_layout.setContentsMargins(10, 0, 10, 10); sidebar_layout.setSpacing(10)

        header_frame = QFrame(); header_frame.setObjectName("header")
        header_layout = QHBoxLayout(header_frame); header_layout.setContentsMargins(10, 10, 10, 10)
        logo_label = QLabel(); pixmap = QPixmap("assets/logo.png")
        logo_label.setPixmap(pixmap.scaled(35, 35, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.company_name_label = QLabel("VALIANCE"); self.company_name_label.setObjectName("company_name")
        header_layout.addWidget(logo_label); header_layout.addWidget(self.company_name_label); header_layout.addStretch()
        sidebar_layout.addWidget(header_frame)

        separator = QFrame(); separator.setFrameShape(QFrame.HLine); separator.setObjectName("separator")
        sidebar_layout.addWidget(separator)

        self.upload_button = QPushButton(QIcon("assets/icons/upload.png"), " Upload Data")
        self.overview_button = QPushButton(QIcon("assets/icons/overview.png"), " Data Overview")
        self.plot_button = QPushButton(QIcon("assets/icons/plot.png"), " Plot Data")
        # --- NEW: AI Analyst Button ---
        self.ai_analyst_button = QPushButton(QIcon("assets/icons/image.png"), " AI Analyst") # Assumes you have an icon

        self.data_ops_button = QToolButton()
        self.data_ops_button.setText(" Data Operations")
        self.data_ops_button.setIcon(QIcon("assets/icons/settings.png"))
        self.data_ops_button.setObjectName("sidebar_button")
        self.data_ops_button.setCheckable(True)
        self.data_ops_button.setPopupMode(QToolButton.InstantPopup)
        data_ops_menu = QMenu(self)
        self.data_ops_actions = {
            2: QAction("Column Selection", self), 3: QAction("Rename & Convert", self),
            4: QAction("Data Cleaner", self), 5: QAction("Aggregation", self)
        }
        for index, action in self.data_ops_actions.items():
            action.triggered.connect(lambda checked, i=index: self.switch_tab(i))
            data_ops_menu.addAction(action)
        self.data_ops_button.setMenu(data_ops_menu)

        self.eda_button = QToolButton()
        self.eda_button.setText(" Exploratory Analysis")
        self.eda_button.setIcon(QIcon("assets/icons/eda.png"))
        self.eda_button.setObjectName("sidebar_button")
        self.eda_button.setCheckable(True)
        self.eda_button.setPopupMode(QToolButton.InstantPopup)
        eda_menu = QMenu(self)
        self.eda_actions = {
            7: QAction("Time Series", self), 8: QAction("Demand Patterns", self),
            9: QAction("Data Quality", self), 10: QAction("Product Insights", self),
            11: QAction("Feature Insights", self), 12: QAction("Volume Analysis", self),
            13: QAction("Find Best Model", self), 14: QAction("Pareto Analysis", self)
        }
        for index, action in self.eda_actions.items():
            action.triggered.connect(lambda checked, i=index: self.switch_tab(i))
            eda_menu.addAction(action)
        self.eda_button.setMenu(eda_menu)

        # --- UPDATED: top_level_buttons to include AI Analyst button ---
        self.top_level_buttons = {
            0: self.upload_button,
            1: self.overview_button,
            6: self.plot_button,
            15: self.ai_analyst_button, # NEW
            -1: self.data_ops_button,
            -2: self.eda_button
        }

        sidebar_layout.addWidget(self.upload_button)
        sidebar_layout.addWidget(self.overview_button)
        sidebar_layout.addWidget(self.data_ops_button)
        sidebar_layout.addWidget(self.plot_button)
        sidebar_layout.addWidget(self.eda_button)
        sidebar_layout.addWidget(self.ai_analyst_button) # Add new button to layout

        for button in self.top_level_buttons.values():
            button.setObjectName("sidebar_button")
            if button != self.upload_button:
                button.setEnabled(False)
        self.upload_button.setChecked(True)

        sidebar_layout.addStretch()

        self.toggle_button = QPushButton()
        self.toggle_button.setObjectName("toggle_button")
        self.toggle_button.setIcon(QIcon("assets/icons/chevron-left.svg"))
        sidebar_layout.addWidget(self.toggle_button, 0, Qt.AlignBottom)

        return sidebar

    def connect_signals(self):
        self.upload_button.clicked.connect(lambda: self.switch_tab(0))
        self.overview_button.clicked.connect(lambda: self.switch_tab(1))
        self.plot_button.clicked.connect(lambda: self.switch_tab(6))
        self.ai_analyst_button.clicked.connect(lambda: self.switch_tab(15)) # Connect new button

        self.app_state.data_loaded.connect(self.on_data_loaded)
        self.app_state.data_cleaned.connect(self.on_data_cleaned)
        self.app_state.data_aggregated.connect(self.on_data_aggregated)

        self.toggle_button.clicked.connect(self.toggle_sidebar)

    def toggle_sidebar(self):
        self.sidebar_is_expanded = not self.sidebar_is_expanded
        end_width = self.SIDEBAR_EXPANDED_WIDTH if self.sidebar_is_expanded else self.SIDEBAR_COLLAPSED_WIDTH
        self.min_animation = QPropertyAnimation(self.sidebar, b"minimumWidth"); self.min_animation.setDuration(300)
        self.min_animation.setStartValue(self.sidebar.width()); self.min_animation.setEndValue(end_width)
        self.min_animation.setEasingCurve(QEasingCurve.InOutCubic); self.min_animation.start()
        self.max_animation = QPropertyAnimation(self.sidebar, b"maximumWidth"); self.max_animation.setDuration(300)
        self.max_animation.setStartValue(self.sidebar.width()); self.max_animation.setEndValue(end_width)
        self.max_animation.setEasingCurve(QEasingCurve.InOutCubic)

        if self.sidebar_is_expanded:
            self.toggle_button.setIcon(QIcon("assets/icons/chevron-left.svg"))
            self.company_name_label.show()
            self.upload_button.setText(" Upload Data"); self.overview_button.setText(" Data Overview")
            self.plot_button.setText(" Plot Data"); self.data_ops_button.setText(" Data Operations")
            self.eda_button.setText(" Exploratory Analysis"); self.ai_analyst_button.setText(" AI Analyst")
        else:
            self.toggle_button.setIcon(QIcon("assets/icons/chevron-right.svg"))
            self.company_name_label.hide()
            for button in self.top_level_buttons.values(): button.setText("")
        self.max_animation.start()

    def on_data_loaded(self, df):
        print("MainWindow: data_loaded signal received.")
        self.overview_button.setEnabled(True)
        self.plot_button.setEnabled(True)
        self.data_ops_button.setEnabled(True)
        self.ai_analyst_button.setEnabled(True) # Enable AI button
        self.eda_button.setEnabled(False)
        
        for i in [1, 2, 3, 4, 5, 6, 15]: # Update all relevant tabs, including AI
            tab = self.content_area.widget(i)
            if hasattr(tab, 'set_data'): tab.set_data(df)
        self.switch_tab(1)

    def on_data_cleaned(self, df):
        print("MainWindow: data_cleaned signal received.")
        for i in [1, 2, 3, 4, 5, 6, 15]: # Update all relevant tabs, including AI
            tab = self.content_area.widget(i)
            if hasattr(tab, 'set_data'): tab.set_data(df)
        self.eda_button.setEnabled(False)

    def on_data_aggregated(self, df):
        print("MainWindow: data_aggregated signal received.")
        self.eda_button.setEnabled(True)
        self.ai_analyst_button.setEnabled(True) # Ensure AI button stays enabled
        
        eda_page_indices = range(7, 15)
        for i in eda_page_indices:
            tab = self.content_area.widget(i)
            if hasattr(tab, 'set_data'): tab.set_data(df)
        self.switch_tab(7)

    def switch_tab(self, index):
        self.content_area.setCurrentIndex(index)
        for btn in self.top_level_buttons.values(): btn.setChecked(False)

        if index in [0, 1, 6, 15]: # Upload, Overview, Plot, AI Analyst
            self.top_level_buttons[index].setChecked(True)
        elif 2 <= index <= 5: # Data Operations
            self.top_level_buttons[-1].setChecked(True)
        elif 7 <= index <= 14: # Exploratory Analysis
            self.top_level_buttons[-2].setChecked(True)

    def load_stylesheet(self):
        try:
            with open('styles/main.qss', 'r') as f: self.setStyleSheet(f.read())
        except FileNotFoundError: print("Warning: styles/main.qss not found.")

    def update_stylesheet(self, stylesheet): self.setStyleSheet(stylesheet)

    def setup_hot_reloader(self):
        self.reloader = StyleReloader()
        self.reloader.update_style_signal.connect(self.update_stylesheet)
        event_handler = QssEventHandler(self.reloader)
        self.observer = Observer()
        self.observer.schedule(event_handler, path='./styles', recursive=False)
        self.observer.start()
        print("--- Stylesheet Hot-reloader is active ---")

    def closeEvent(self, event):
        self.observer.stop(); self.observer.join(); event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv); window = MainWindow(); window.show(); sys.exit(app.exec_())
