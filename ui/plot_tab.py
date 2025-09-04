# ui/plot_tab.py

import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QFrame, QLabel,
                             QComboBox, QHBoxLayout, QSizePolicy, QLineEdit,
                             QScrollArea, QDialog, QListWidget, QListWidgetItem,
                             QGroupBox, QGridLayout)
from PyQt5.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QIcon

# --- Import for Plotly and Web Engine ---
from PyQt5.QtWebEngineWidgets import QWebEngineView
import plotly.express as px
import plotly.io as pio

from app_state import AppState

# Set a professional default plotly theme
pio.templates.default = "plotly_white"


# --- Dynamic Filter Widget ---
class FilterWidget(QFrame):
    filter_changed = pyqtSignal()

    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.df = df
        self.setObjectName("filterCard")
        self.setStyleSheet("""
            #filterCard {
                background-color: #f8f9fa;
                border-radius: 5px;
                border: 1px solid #dee2e6;
                padding: 5px;
            }
        """)

        self.main_layout = QHBoxLayout(self)
        self.main_layout.setSpacing(8)
        self.main_layout.setContentsMargins(5, 5, 5, 5)

        self.column_combo = QComboBox()
        self.column_combo.setMinimumWidth(120)
        self.column_combo.addItems(df.columns)
        self.column_combo.currentIndexChanged.connect(self.update_filter_type)

        self.remove_button = QPushButton("✖")
        self.remove_button.setFixedSize(25, 25)
        self.remove_button.setObjectName("removeButton")

        self.main_layout.addWidget(self.column_combo, 3)

        self.filter_control_widget = QWidget()
        self.filter_control_layout = QHBoxLayout(self.filter_control_widget)
        self.filter_control_layout.setContentsMargins(0, 0, 0, 0)
        self.filter_control_layout.setSpacing(5)
        self.main_layout.addWidget(self.filter_control_widget, 5)
        self.main_layout.addWidget(self.remove_button, 1)

        self.update_filter_type()

    def update_filter_type(self):
        while self.filter_control_layout.count():
            child = self.filter_control_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        col_name = self.column_combo.currentText()
        if not col_name:
            return

        dtype = self.df[col_name].dtype

        self.op_combo = QComboBox()
        self.min_edit = QLineEdit()
        self.max_edit = QLineEdit()
        self.value_button = QPushButton("Select Values")

        if pd.api.types.is_numeric_dtype(dtype):
            self.op_combo.addItems(["between", ">", "<", ">=", "<=", "==", "!="])
            self.min_edit.setPlaceholderText("Value")
            self.max_edit.setPlaceholderText("Max (for between)")
            self.filter_control_layout.addWidget(self.op_combo, 2)
            self.filter_control_layout.addWidget(self.min_edit, 3)
            self.and_label = QLabel("and")
            self.filter_control_layout.addWidget(self.and_label, 1)
            self.filter_control_layout.addWidget(self.max_edit, 3)
            self.op_combo.currentIndexChanged.connect(self._toggle_numeric_inputs)
            self._toggle_numeric_inputs()

        elif pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_object_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
            self.op_combo.addItems(["is in", "not in"])
            self.value_button.setMinimumWidth(120)
            self.value_button.clicked.connect(self.open_value_selector)
            self.selected_values = list(self.df[col_name].unique())
            self.filter_control_layout.addWidget(self.op_combo, 2)
            self.filter_control_layout.addWidget(self.value_button, 5)

        elif pd.api.types.is_datetime64_any_dtype(dtype):
            self.op_combo.addItems(["between", ">", "<", ">=", "<="])
            self.min_edit.setPlaceholderText("YYYY-MM-DD")
            self.max_edit.setPlaceholderText("YYYY-MM-DD")
            self.filter_control_layout.addWidget(self.op_combo, 2)
            self.filter_control_layout.addWidget(self.min_edit, 3)
            self.and_label = QLabel("and")
            self.filter_control_layout.addWidget(self.and_label, 1)
            self.filter_control_layout.addWidget(self.max_edit, 3)
            self.op_combo.currentIndexChanged.connect(self._toggle_numeric_inputs)
            self._toggle_numeric_inputs()

    def _toggle_numeric_inputs(self):
        is_between = self.op_combo.currentText() == "between"
        self.and_label.setVisible(is_between)
        self.max_edit.setVisible(is_between)

    def open_value_selector(self):
        col_name = self.column_combo.currentText()
        unique_vals = self.df[col_name].dropna().unique()
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Select values for '{col_name}'")
        dialog.setMinimumSize(300, 400)
        layout = QVBoxLayout(dialog)
        list_widget = QListWidget()
        list_widget.setSelectionMode(QListWidget.MultiSelection)
        for val in unique_vals:
            item = QListWidgetItem(str(val))
            list_widget.addItem(item)
            if str(val) in [str(v) for v in self.selected_values]:
                item.setSelected(True)
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.setObjectName("okButton")
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addStretch()
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(ok_button)
        ok_button.clicked.connect(dialog.accept)
        layout.addWidget(QLabel(f"<b>Select values for: {col_name}</b>"))
        layout.addWidget(list_widget, 1)
        layout.addLayout(button_layout)
        if dialog.exec_():
            self.selected_values = [item.text() for item in list_widget.selectedItems()]
            self.value_button.setText(f"{len(self.selected_values)} values selected")
            self.filter_changed.emit()

    def get_filter(self):
        col_name = self.column_combo.currentText()
        if not col_name:
            return None
        op = self.op_combo.currentText()
        try:
            if pd.api.types.is_numeric_dtype(self.df[col_name].dtype):
                if op == "between":
                    min_val = float(self.min_edit.text()) if self.min_edit.text() else -np.inf
                    max_val = float(self.max_edit.text()) if self.max_edit.text() else np.inf
                    return lambda df: df[col_name].between(min_val, max_val)
                else:
                    if not self.min_edit.text():
                        return None
                    val = float(self.min_edit.text())
                    return {
                        ">": lambda df: df[col_name] > val,
                        "<": lambda df: df[col_name] < val,
                        ">=": lambda df: df[col_name] >= val,
                        "<=": lambda df: df[col_name] <= val,
                        "==": lambda df: df[col_name] == val,
                        "!=": lambda df: df[col_name] != val,
                    }.get(op)
            elif pd.api.types.is_datetime64_any_dtype(self.df[col_name].dtype):
                val = pd.to_datetime(self.min_edit.text(), errors='coerce')
                if pd.isna(val):
                    return None
                if op == "between":
                    max_val = pd.to_datetime(self.max_edit.text(), errors='coerce')
                    if pd.isna(max_val):
                        return None
                    return lambda df: df[col_name].between(val, max_val)
                else:
                    return {
                        ">": lambda df: df[col_name] > val,
                        "<": lambda df: df[col_name] < val,
                        ">=": lambda df: df[col_name] >= val,
                        "<=": lambda df: df[col_name] <= val,
                    }.get(op)
            else:
                vals_to_check = [str(v) for v in self.selected_values]
                if op == "is in":
                    return lambda df: df[col_name].astype(str).isin(vals_to_check)
                if op == "not in":
                    return lambda df: ~df[col_name].astype(str).isin(vals_to_check)
        except Exception as e:
            print(f"Filter error: {e}")
            return None


# --- Main Plotting Tab ---
class PlotTab(QWidget):
    def __init__(self, app_state: AppState):
        super().__init__()
        self.app_state = app_state
        self.df = None
        self.current_source_name = ""
        self.filter_widgets = []
        self.is_sidebar_visible = True

        # --- Main Layout ---
        self.page_layout = QHBoxLayout(self)
        self.page_layout.setContentsMargins(15, 15, 15, 15)
        self.page_layout.setSpacing(5)

        # --- Controls Sidebar ---
        self.controls_sidebar = QFrame()
        self.controls_sidebar.setObjectName("card")
        self.controls_sidebar.setMinimumWidth(350)
        self.controls_sidebar.setMaximumWidth(450)

        sidebar_main_layout = QVBoxLayout(self.controls_sidebar)
        sidebar_main_layout.setContentsMargins(10, 10, 10, 10)
        sidebar_main_layout.setSpacing(15)
        sidebar_main_layout.setAlignment(Qt.AlignTop)

        title = QLabel("Visualization Dashboard")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        sidebar_main_layout.addWidget(title)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")

        scroll_widget = QWidget()
        self.controls_layout = QVBoxLayout(scroll_widget)
        self.controls_layout.setAlignment(Qt.AlignTop)
        self.controls_layout.setSpacing(15)

        self.setup_control_widgets(self.controls_layout)

        scroll_area.setWidget(scroll_widget)
        sidebar_main_layout.addWidget(scroll_area, 1)

        self.generate_button = QPushButton("Generate / Update Plot")
        self.generate_button.setObjectName("generateButton")
        self.generate_button.clicked.connect(self.generate_plot)
        sidebar_main_layout.addWidget(self.generate_button)

        # --- Retractable Sidebar Toggle Button ---
        self.toggle_button = QPushButton("◀")
        self.toggle_button.setObjectName("toggle_button_plot")
        self.toggle_button.setFixedSize(20, 40)
        self.toggle_button.clicked.connect(self.toggle_sidebar)

        # --- Main Content Area ---
        main_content_area = QFrame()
        main_content_area.setObjectName("main_plot_area")
        main_content_layout = QVBoxLayout(main_content_area)
        main_content_layout.setContentsMargins(5, 5, 5, 5)

        self.plotly_widget = QWebEngineView()
        main_content_layout.addWidget(self.plotly_widget, 1)

        # Add to main layout
        self.page_layout.addWidget(self.controls_sidebar)
        self.page_layout.addWidget(self.toggle_button)
        self.page_layout.addWidget(main_content_area, 1)

        # Connect signals
        self.app_state.data_loaded.connect(self.update_available_sources)
        self.app_state.data_cleaned.connect(self.update_available_sources)
        self.app_state.data_aggregated.connect(self.update_available_sources)

        self.update_available_sources()
        self.show_message("Please select a data source to begin visualizing.")

        # Apply global styling
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            #card, #main_plot_area {
                background-color: #ffffff;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
            }
            #main_plot_area {
                background-color: #f9f9f9;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 5px;
                background-color: #fafafa;
            }
            QGroupBox::title {
                subline-offset: -6px;
                padding: 0 8px;
                color: #333;
            }
            QPushButton {
                padding: 6px 10px;
                border-radius: 4px;
                background-color: #007acc;
                color: white;
                border: none;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
            QPushButton#removeButton {
                background-color: #dc3545;
                font-weight: bold;
                padding: 0;
            }
            QPushButton#removeButton:hover {
                background-color: #c82333;
            }
            QPushButton#okButton {
                background-color: #28a745;
            }
            QPushButton#okButton:hover {
                background-color: #218838;
            }
            QPushButton#generateButton {
                background-color: #28a745;
                font-weight: bold;
                padding: 10px;
                font-size: 13px;
            }
            QPushButton#generateButton:hover {
                background-color: #218838;
            }
            QComboBox, QLineEdit {
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: #fdfdfd;
            }
            QScrollArea {
                border: none;
            }
            QLabel {
                color: #333;
            }
            #toggle_button_plot {
                background-color: #6c757d;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                border: none;
            }
            #toggle_button_plot:hover {
                background-color: #5a6268;
            }
        """)

    def set_data(self, df: pd.DataFrame):
        """Compatibility method called by main.py."""
        # Save current selection before refresh
        current_source = self.df_select_combo.currentText()
        self.app_state.raw_df = df
        self.app_state.cleaned_df = df.copy()
        self.app_state.aggregated_df = None

        # Refresh the source list
        self.update_available_sources()

        # Force re-select current source to load the data
        if current_source and self.df_select_combo.findText(current_source) != -1:
            self.df_select_combo.setCurrentText(current_source)
            self._on_df_source_changed()  # ⚠️ Critical: Manually trigger reload

    def setup_control_widgets(self, layout):
        source_group = QGroupBox("1. Data Source")
        source_layout = QVBoxLayout(source_group)
        self.df_select_combo = QComboBox()
        self.df_select_combo.currentIndexChanged.connect(self._on_df_source_changed)
        source_layout.addWidget(self.df_select_combo)

        plot_config_group = QGroupBox("2. Plot Configuration")
        plot_config_layout = QGridLayout(plot_config_group)
        plot_config_layout.setSpacing(8)
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Line", "Bar", "Scatter", "Histogram", "Box Plot", "Density Heatmap"])
        self.plot_type_combo.currentIndexChanged.connect(self._update_plot_controls)
        self.x_axis_combo = QComboBox()
        self.y_axis_combo = QComboBox()
        self.hue_combo = QComboBox()
        self.y_axis_label = QLabel("Y-Axis:")
        self.hue_label = QLabel("Color By:")

        plot_config_layout.addWidget(QLabel("Plot Type:"), 0, 0)
        plot_config_layout.addWidget(self.plot_type_combo, 0, 1)
        plot_config_layout.addWidget(QLabel("X-Axis:"), 1, 0)
        plot_config_layout.addWidget(self.x_axis_combo, 1, 1)
        plot_config_layout.addWidget(self.y_axis_label, 2, 0)
        plot_config_layout.addWidget(self.y_axis_combo, 2, 1)
        plot_config_layout.addWidget(self.hue_label, 3, 0)
        plot_config_layout.addWidget(self.hue_combo, 3, 1)

        filter_group = QGroupBox("3. Data Filters")
        filter_area_layout = QVBoxLayout(filter_group)
        self.filter_widgets_layout = QVBoxLayout()
        filter_widgets_container = QWidget()
        filter_widgets_container.setLayout(self.filter_widgets_layout)
        filter_scroll = QScrollArea()
        filter_scroll.setWidgetResizable(True)
        filter_scroll.setWidget(filter_widgets_container)
        filter_scroll.setMinimumHeight(100)

        filter_button_layout = QHBoxLayout()
        self.add_filter_button = QPushButton("Add Filter")
        self.add_filter_button.clicked.connect(self.add_filter)
        self.apply_filters_button = QPushButton("Apply Filters")
        self.apply_filters_button.clicked.connect(self.generate_plot)
        filter_button_layout.addWidget(self.add_filter_button)
        filter_button_layout.addWidget(self.apply_filters_button)

        filter_area_layout.addLayout(filter_button_layout)
        filter_area_layout.addWidget(filter_scroll, 1)

        layout.addWidget(source_group)
        layout.addWidget(plot_config_group)
        layout.addWidget(filter_group)

    def toggle_sidebar(self):
        self.is_sidebar_visible = not self.is_sidebar_visible
        self.animation = QPropertyAnimation(self.controls_sidebar, b"maximumWidth")
        self.animation.setDuration(250)
        self.animation.setEasingCurve(QEasingCurve.InOutCubic)
        if self.is_sidebar_visible:
            self.animation.setStartValue(0)
            self.animation.setEndValue(450)
            self.toggle_button.setText("◀")
        else:
            self.animation.setStartValue(self.controls_sidebar.width())
            self.animation.setEndValue(0)
            self.toggle_button.setText("▶")
        self.animation.start()

    def update_available_sources(self, df=None):
        self.df_select_combo.blockSignals(True)
        current_selection = self.df_select_combo.currentText()
        self.df_select_combo.clear()
        sources = [""] 
        if self.app_state.raw_df is not None: sources.append("Raw Data")
        if self.app_state.cleaned_df is not None: sources.append("Cleaned Data")
        if self.app_state.aggregated_df is not None: sources.append("Aggregated Data")
        
        self.df_select_combo.addItems(sources)

        # Restore selection
        if current_selection in sources:
            self.df_select_combo.setCurrentText(current_selection)
            self.df_select_combo.blockSignals(False)
            # ⚠️ Force emit the signal if selection is valid
            if current_selection:
                self._on_df_source_changed()
        else:
            self.df_select_combo.blockSignals(False)
            if self.df_select_combo.count() > 1:
                self.df_select_combo.setCurrentIndex(1)  # Auto-select "Raw Data"
                self._on_df_source_changed()

        # Show message if no source
        if self.df_select_combo.currentText() == "" and self.df is not None:
            self.df = None
            self._update_plot_controls()
            self.show_message("Please select a data source to visualize.")

    def _on_df_source_changed(self):
        source_name = self.df_select_combo.currentText()
        if not source_name:
            self.df = None
            self._update_plot_controls()
            self.show_message("Please select a data source to visualize")
            return
        if source_name == self.current_source_name:
            return
        self.current_source_name = source_name
        df_map = {
            "Raw Data": self.app_state.raw_df,
            "Cleaned Data": self.app_state.cleaned_df,
            "Aggregated Data": self.app_state.aggregated_df
        }
        self.df = df_map.get(source_name)
        self.clear_filters()
        self._update_plot_controls()
        self.generate_plot()

    def _update_plot_controls(self):
        for combo in [self.x_axis_combo, self.y_axis_combo, self.hue_combo]:
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("None")
        if self.df is None:
            return

        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        datetime_cols = self.df.select_dtypes(include=['datetime64[ns]']).columns.tolist()

        plot_type = self.plot_type_combo.currentText()

        if plot_type in ["Scatter", "Line"]:
            self.x_axis_combo.addItems(datetime_cols + numeric_cols)
            self.y_axis_combo.addItems(numeric_cols)
        elif plot_type in ["Bar", "Box Plot"]:
            self.x_axis_combo.addItems(categorical_cols + datetime_cols)
            self.y_axis_combo.addItems(numeric_cols)
        elif plot_type == "Histogram":
            self.x_axis_combo.addItems(numeric_cols)
        elif plot_type == "Density Heatmap":
            self.x_axis_combo.addItems(numeric_cols + datetime_cols)
            self.y_axis_combo.addItems(numeric_cols)
        else:
            self.x_axis_combo.addItems(self.df.columns)

        self.hue_combo.addItems(categorical_cols + numeric_cols)

        # Set defaults
        if datetime_cols and plot_type in ["Line", "Scatter"]:
            self.x_axis_combo.setCurrentText(datetime_cols[0])
        elif categorical_cols and plot_type in ["Bar", "Box Plot"]:
            self.x_axis_combo.setCurrentText(categorical_cols[0])
        elif self.x_axis_combo.count() > 1:
            self.x_axis_combo.setCurrentIndex(1)

        if plot_type != "Histogram" and len(numeric_cols) > 0:
            self.y_axis_combo.setCurrentText(numeric_cols[0])

        for combo in [self.x_axis_combo, self.y_axis_combo, self.hue_combo]:
            combo.blockSignals(False)

        is_hist = plot_type == "Histogram"
        is_heatmap = plot_type == "Density Heatmap"
        self.y_axis_label.setVisible(not is_hist)
        self.y_axis_combo.setVisible(not is_hist)
        self.hue_label.setVisible(not is_heatmap)
        self.hue_combo.setVisible(not is_heatmap)

    def add_filter(self):
        if self.df is None:
            return
        fw = FilterWidget(self.df)
        fw.remove_button.clicked.connect(lambda: self.remove_filter(fw))
        self.filter_widgets_layout.addWidget(fw)
        self.filter_widgets.append(fw)

    def remove_filter(self, filter_widget):
        self.filter_widgets.remove(filter_widget)
        filter_widget.deleteLater()

    def clear_filters(self):
        for fw in self.filter_widgets:
            fw.deleteLater()
        self.filter_widgets.clear()

    def apply_filters(self):
        if self.df is None:
            return pd.DataFrame()
        filtered_df = self.df.copy()
        for fw in self.filter_widgets:
            filter_func = fw.get_filter()
            if filter_func:
                try:
                    filtered_df = filtered_df.loc[filter_func(filtered_df)]
                except Exception as e:
                    print(f"Error applying filter: {e}")
        return filtered_df

    def show_message(self, text, is_error=False):
        color = "#dc3545" if is_error else "#6c757d"
        font = "Arial, sans-serif"
        html = f"""
        <html>
        <head>
            <style>
                body {{ 
                    font-family: '{font}';
                    background-color: #f8f9fa;
                    margin: 0;
                    padding: 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    color: {color};
                }}
                .container {{
                    text-align: center;
                    max-width: 500px;
                    padding: 30px;
                    background-color: white;
                    border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }}
                h2 {{
                    margin: 0 0 16px 0;
                    font-size: 18px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>{text}</h2>
            </div>
        </body>
        </html>
        """
        self.plotly_widget.setHtml(html)

    def generate_plot(self):
        if self.df is None or self.df.empty:
            self.show_message("No data loaded. Please select a valid data source.")
            return

        filtered_df = self.apply_filters()
        if filtered_df.empty:
            self.show_message("No data matches the current filters.")
            return

        plot_type = self.plot_type_combo.currentText()
        x_col = self.x_axis_combo.currentText()
        y_col = self.y_axis_combo.currentText()
        hue_col = self.hue_combo.currentText()

        if x_col == "None":
            self.show_message("Please select an X-axis column.")
            return
        if plot_type not in ["Histogram", "Density Heatmap"] and y_col == "None":
            self.show_message("This plot type requires a Y-axis column.")
            return

        try:
            fig = None
            color_arg = hue_col if hue_col != "None" else None

            if plot_type == "Histogram":
                fig = px.histogram(
                    filtered_df,
                    x=x_col,
                    color=color_arg,
                    title=f"Histogram of {x_col}" + (f" (colored by {hue_col})" if color_arg else ""),
                    hover_data=filtered_df.columns.tolist()
                )
            elif plot_type == "Density Heatmap":
                if y_col == "None":
                    self.show_message("Density Heatmap requires both X and Y axes.")
                    return
                fig = px.density_heatmap(
                    filtered_df,
                    x=x_col,
                    y=y_col,
                    color_continuous_scale="Viridis",
                    title=f"Density Heatmap: {x_col} vs {y_col}",
                    histfunc='count'
                )
                if color_arg:
                    fig.update_layout(legend_title_text=color_arg)
            else:
                kwargs = {
                    'x': x_col,
                    'data_frame': filtered_df,
                    'title': f"{plot_type} Plot: {y_col if y_col != 'None' else x_col} vs {x_col}",
                    'hover_data': filtered_df.columns if plot_type == "Scatter" else None
                }
                if y_col != "None":
                    kwargs['y'] = y_col
                if color_arg:
                    kwargs['color'] = color_arg
                if plot_type == "Line":
                    kwargs['markers'] = True
                elif plot_type == "Box Plot":
                    kwargs['points'] = "outliers"

                plot_func_name = plot_type.lower().replace(" ", "_")
                if not hasattr(px, plot_func_name):
                    self.show_message(f"Plot type '{plot_type}' is not supported.")
                    return
                fig = getattr(px, plot_func_name)(**kwargs)

            if fig:
                fig.update_layout(
                    title_x=0.5,
                    margin=dict(l=50, r=30, b=60, t=80),
                    plot_bgcolor="white",
                    legend_title_text=color_arg or "Category",
                    font=dict(family="Arial, sans-serif", size=12),
                    hovermode="closest"
                )
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#eee', zeroline=False)
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#eee', zeroline=False)

                # ✅ Fixed: use 'showlogo', not 'displayLogo', and use 'inline' for reliability
                html = fig.to_html(
                    include_plotlyjs='inline',  # ✅ Ensures plot works offline
                    full_html=True,
                    config={
                        'displayModeBar': True,
                        'scrollZoom': True,
                        'showlogo': False,  # ✅ Correct key
                        'edits': {'legendPosition': True}
                    }
                )
                self.plotly_widget.setHtml(html)
            else:
                self.show_message("Could not generate plot.")

        except Exception as e:
            self.show_message(f"Error: {str(e)}", is_error=True)
            print(f"Plot Generation Error: {e}")