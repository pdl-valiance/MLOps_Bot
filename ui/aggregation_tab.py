# ui/aggregation_tab.py

import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QFrame, QLabel,
                             QScrollArea, QListWidget, QListWidgetItem, QFileDialog,
                             QComboBox, QHBoxLayout, QHeaderView, QTableWidget, QTableWidgetItem,
                             QGridLayout)
from PyQt5.QtCore import Qt
from app_state import AppState

class AggregationTab(QWidget):
    def __init__(self, app_state: AppState):
        super().__init__()
        self.app_state = app_state

        self.AGG_OPTIONS = ['sum', 'mean', 'median', 'min', 'max', 'std', 'count', 'nunique', 'first', 'last']

        # --- Main Layout with Scroll Area ---
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")
        page_layout.addWidget(scroll_area)
        
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        
        main_layout = QVBoxLayout(content_widget)
        main_layout.setContentsMargins(25, 25, 25, 25)
        main_layout.setSpacing(15)
        main_layout.setAlignment(Qt.AlignTop)

        # Title and description
        title = QLabel("Hierarchy Grouping & Aggregation")
        title.setObjectName("h1_label")
        description = QLabel("Select dimensions to group by and define how to aggregate numeric values. This creates the time series data for analysis.")
        description.setWordWrap(True)
        main_layout.addWidget(title)
        main_layout.addWidget(description)

        # Configuration Layout
        config_layout = QHBoxLayout()
        config_layout.setSpacing(20)

        # Left Pane: Dimension Selection
        dim_card = QFrame(); dim_card.setObjectName("card")
        dim_layout = QVBoxLayout(dim_card)
        dim_title = QLabel("Step 1: Select Dimensions"); dim_title.setObjectName("h2_label")
        self.dim_list = QListWidget()
        self.dim_list.itemChanged.connect(self.update_selection_summary)
        dim_layout.addWidget(dim_title)
        dim_layout.addWidget(self.dim_list)

        # Right Pane: Aggregation Method Selection
        agg_card = QFrame(); agg_card.setObjectName("card")
        agg_layout = QVBoxLayout(agg_card)
        agg_title = QLabel("Step 2: Define Aggregations"); agg_title.setObjectName("h2_label")
        agg_scroll_area = QScrollArea(); agg_scroll_area.setWidgetResizable(True)
        agg_scroll_area.setStyleSheet("QScrollArea { border: none; background-color: white; }")
        agg_selectors_widget = QWidget()
        self.agg_selectors_layout = QGridLayout(agg_selectors_widget)
        self.agg_selectors_layout.setAlignment(Qt.AlignTop)
        self.agg_selectors_layout.setSpacing(15)
        agg_scroll_area.setWidget(agg_selectors_widget)
        agg_layout.addWidget(agg_title)
        agg_layout.addWidget(agg_scroll_area)

        config_layout.addWidget(dim_card, 1)
        config_layout.addWidget(agg_card, 1)
        main_layout.addLayout(config_layout)

        # Action Bar
        action_layout = QHBoxLayout()
        self.selection_summary_label = QLabel("Grouping by: Date")
        self.selection_summary_label.setStyleSheet("font-weight: 500; color: #495057; padding: 5px;")
        self.agg_button = QPushButton("Aggregate Data"); self.agg_button.clicked.connect(self.apply_aggregation)
        action_layout.addWidget(self.selection_summary_label, 1)
        action_layout.addWidget(self.agg_button)
        main_layout.addLayout(action_layout)

        # Preview Area
        self.preview_card = QFrame(); self.preview_card.setObjectName("card")
        self.preview_card.setVisible(False)
        preview_layout = QVBoxLayout(self.preview_card)
        preview_title = QLabel("Aggregated Data Preview"); preview_title.setObjectName("h2_label")
        self.preview_table = QTableWidget()
        self.download_button = QPushButton("Download Aggregated Data"); self.download_button.clicked.connect(self.download_data)
        
        preview_controls_layout = QHBoxLayout()
        preview_controls_layout.addStretch()
        preview_controls_layout.addWidget(self.download_button)
        
        preview_layout.addWidget(preview_title)
        preview_layout.addWidget(self.preview_table)
        preview_layout.addLayout(preview_controls_layout)
        main_layout.addWidget(self.preview_card)

    def set_data(self, df: pd.DataFrame):
        if df is None: return
        self.df = df
        self.preview_card.setVisible(False)
        
        self.dim_list.clear()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        date_col = next((c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])), None)
        for col in categorical_cols:
            if col != date_col:
                item = QListWidgetItem(col)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
                self.dim_list.addItem(item)
            
        self.agg_widgets = {}
        while self.agg_selectors_layout.count():
            child = self.agg_selectors_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()

        numeric_cols = df.select_dtypes(include=['number']).columns
        for i, col in enumerate(numeric_cols):
            label = QLabel(f"'{col}':"); combo = QComboBox(); combo.addItems(self.AGG_OPTIONS)
            if any(k in col.lower() for k in ['price', 'rate', 'avg']): combo.setCurrentText('mean')
            else: combo.setCurrentText('sum')
            self.agg_selectors_layout.addWidget(label, i, 0)
            self.agg_selectors_layout.addWidget(combo, i, 1)
            self.agg_widgets[col] = combo
        self.update_selection_summary()

    def update_selection_summary(self):
        selected_dims = [self.dim_list.item(i).text() for i in range(self.dim_list.count()) if self.dim_list.item(i).checkState() == Qt.Checked]
        if selected_dims: self.selection_summary_label.setText(f"<b>Grouping by:</b> Date, {', '.join(selected_dims)}")
        else: self.selection_summary_label.setText("<b>Grouping by:</b> Date")

    def apply_aggregation(self):
        df = self.app_state.cleaned_df
        date_col = next((c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])), None)
        if df is None or date_col is None: print("Aggregation Error: Data or Date column not found"); return

        selected_dims = [self.dim_list.item(i).text() for i in range(self.dim_list.count()) if self.dim_list.item(i).checkState() == Qt.Checked]
        group_by_cols = [date_col] + selected_dims
        agg_funcs = {col: combo.currentText() for col, combo in self.agg_widgets.items()}
        
        try:
            aggregated_df = df.groupby(group_by_cols, as_index=False).agg(agg_funcs)
            self.app_state.update_aggregated_data(aggregated_df, selected_dims)
            self.display_preview(aggregated_df)
        except Exception as e:
            print(f"Aggregation Error: {e}"); self.preview_card.setVisible(False)

    def display_preview(self, df: pd.DataFrame):
        self.preview_table.clear(); self.preview_table.setRowCount(len(df)); self.preview_table.setColumnCount(len(df.columns))
        self.preview_table.setHorizontalHeaderLabels(df.columns)
        
        # Enable the table's scrollbar
        self.preview_table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        for r in range(len(df)):
            for c, col_name in enumerate(df.columns):
                self.preview_table.setItem(r, c, QTableWidgetItem(str(df.iat[r, c])))
        
        self.preview_table.resizeColumnsToContents()
        self.preview_table.resizeRowsToContents()

        # --- THIS BLOCK SETS THE VISIBLE ROW LIMIT TO 20 ---
        rows_to_display = 10
        if self.preview_table.rowCount() > rows_to_display:
            header_height = self.preview_table.horizontalHeader().height()
            total_rows_height = sum(self.preview_table.rowHeight(i) for i in range(rows_to_display))
            
            # Set a fixed height to enable the table's internal scrollbar
            self.preview_table.setFixedHeight(header_height + total_rows_height)
        else:
            # If fewer rows, let it take its natural size
            self.preview_table.setMaximumHeight(16777215) # Remove any previous max height limit
            self.preview_table.setMinimumHeight(0) # Ensure it can shrink
            self.preview_table.adjustSize() # Adjust to content

        self.preview_card.setVisible(True)

    def download_data(self):
        if self.app_state.aggregated_df is None: return
        
        path, _ = QFileDialog.getSaveFileName(self, "Save Aggregated Data", "aggregated_data.csv", "CSV Files (*.csv)")
        if path:
            try:
                self.app_state.aggregated_df.to_csv(path, index=False); print(f"Data saved to {path}")
            except Exception as e:
                print(f"Error saving file: {e}")