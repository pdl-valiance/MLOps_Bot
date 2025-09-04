# ui/overview_tab.py

import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QFrame, QLabel, QTableWidget, 
                             QTableWidgetItem, QGridLayout, QHeaderView, QScrollArea, 
                             QHBoxLayout, QLineEdit)
from PyQt5.QtCore import Qt

class DataOverviewTab(QWidget):
    def __init__(self):
        super().__init__()
        
        # This structure enables the main page scrollbar
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")
        page_layout.addWidget(scroll_area)
        
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        
        main_layout = QVBoxLayout(content_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.setAlignment(Qt.AlignTop) # This keeps content pushed to the top

        # Title
        title = QLabel("Data Overview & Summary")
        title.setObjectName("h1_label")
        main_layout.addWidget(title)
        
        # Stats Card
        stats_card = QFrame()
        stats_card.setObjectName("card")
        stats_layout = QVBoxLayout(stats_card)
        stats_title = QLabel("Dataset Quick Stats")
        stats_title.setObjectName("h2_label")
        stats_layout.addWidget(stats_title)
        self.stats_grid = QGridLayout()
        self.stats_grid.setSpacing(15)
        stats_layout.addLayout(self.stats_grid)
        main_layout.addWidget(stats_card)
        
        # Column Details Card
        details_card = QFrame()
        details_card.setObjectName("card")
        details_layout = QVBoxLayout(details_card)
        details_title = QLabel("Column Details")
        details_title.setObjectName("h2_label")
        details_layout.addWidget(details_title)
        
        self.filter_input = QLineEdit()
        self.filter_input.setObjectName("filter_bar")
        self.filter_input.setPlaceholderText("Search for a column name...")
        self.filter_input.setMaximumWidth(200)
        self.filter_input.textChanged.connect(self.filter_table)
        details_layout.addWidget(self.filter_input)
        
        self.column_table = QTableWidget()
        details_layout.addWidget(self.column_table)
        main_layout.addWidget(details_card)

    def set_data(self, df: pd.DataFrame):
        if df is None:
            self.clear_view()
            return
        self.update_quick_stats(df)
        self.update_column_details(df)
    
    def clear_view(self):
        self.update_quick_stats(None)
        self.column_table.setRowCount(0)

    def update_quick_stats(self, df):
        for i in reversed(range(self.stats_grid.count())):
            widget = self.stats_grid.itemAt(i).widget()
            if widget: widget.setParent(None)
        if df is None: return
            
        num_rows, num_cols = df.shape
        total_missing = df.isnull().sum().sum()
        total_cells = np.prod(df.shape)
        missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0
        duplicate_rows = df.duplicated().sum()
        duplicate_pct = (duplicate_rows / num_rows * 100) if num_rows > 0 else 0
        
        stats = {
            "Rows": f"{num_rows:,}", "Columns": f"{num_cols:,}",
            "Missing Values": f"{total_missing:,} ({missing_pct:.2f}%)",
            "Duplicate Rows": f"{duplicate_rows:,} ({duplicate_pct:.2f}%)"
        }
        for row, (name, value) in enumerate(stats.items()):
            name_label = QLabel(name); name_label.setObjectName("stats_label")
            value_label = QLabel(str(value)); value_label.setObjectName("stats_value_label")
            self.stats_grid.addWidget(name_label, row, 0)
            self.stats_grid.addWidget(value_label, row, 1, alignment=Qt.AlignRight)

    def update_column_details(self, df):
        self.column_table.setRowCount(df.shape[1])
        self.column_table.setColumnCount(5)
        self.column_table.setHorizontalHeaderLabels(["Column", "Data Type", "Missing", "Cardinality", "Statistics / Details"])
        self.column_table.verticalHeader().setVisible(False)
        self.column_table.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.column_table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # --- CHANGE 1: Set default alignment for the header text ---
        self.column_table.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        
        # --- CHANGE 2: Set a minimum width for columns to prevent them from being too narrow ---
        self.column_table.horizontalHeader().setMinimumSectionSize(150)


        for i, col in enumerate(df.columns):
            s = df[col]
            self.column_table.setItem(i, 0, self.create_aligned_item(col))
            self.column_table.setItem(i, 1, self.create_aligned_item(str(s.dtype)))
            missing_count = s.isnull().sum(); missing_pct = (missing_count / len(df) * 100) if len(df) > 0 else 0
            self.column_table.setItem(i, 2, self.create_aligned_item(f"{missing_count:,} ({missing_pct:.2f}%)"))
            self.column_table.setCellWidget(i, 3, self.create_cardinality_widget(s))
            self.column_table.setCellWidget(i, 4, self.create_stats_label(s))

        self.column_table.resizeColumnsToContents()
        # --- THIS IS THE CHANGE ---
        # Instead of stretching only the last column, we now stretch all of them equally.
        header = self.column_table.horizontalHeader()
        for i in range(header.count()):
            header.setSectionResizeMode(i, QHeaderView.Stretch)
        
        self.column_table.resizeRowsToContents()
        
        # --- FINAL HEIGHT LOGIC ---
        rows_to_display = 10
        if self.column_table.rowCount() > rows_to_display:
            header_height = self.column_table.horizontalHeader().height()
            total_rows_height = sum(self.column_table.rowHeight(i) for i in range(rows_to_display))
            self.column_table.setFixedHeight(header_height + total_rows_height)
        else:
            self.column_table.setMaximumHeight(16777215)
            self.column_table.setMinimumHeight(0)
            self.column_table.adjustSize()


    def create_aligned_item(self, text):
        item = QTableWidgetItem(str(text))
        # item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        # This line sets the alignment for each cell's content
        item.setTextAlignment(Qt.AlignLeft)
        return item

    def create_cardinality_widget(self, series):
        unique_count = series.nunique()
        widget = QWidget(); layout = QHBoxLayout(widget)
        layout.setContentsMargins(0,0,0,0); layout.setAlignment(Qt.AlignCenter)
        if unique_count == 1: label = QLabel(f"Constant ({unique_count})"); label.setObjectName("chip_low")
        elif unique_count <= 25 and not pd.api.types.is_numeric_dtype(series.dtype): label = QLabel(f"Low ({unique_count})"); label.setObjectName("chip_low")
        elif (unique_count / len(series) < 0.1) if len(series)>0 else False: label = QLabel(f"Medium ({unique_count})"); label.setObjectName("chip_medium")
        else: label = QLabel(f"High ({unique_count})"); label.setObjectName("chip_high")
        layout.addWidget(label)
        return widget

    def create_stats_label(self, series):
        """Creates a container widget for the stats label to ensure proper resizing."""
        if pd.api.types.is_numeric_dtype(series.dtype):
            stats_str = f"Mean: {series.mean():.2f}\nMedian: {series.median():.2f}\nStd Dev: {series.std():.2f}\nMin: {series.min():.2f}\nMax: {series.max():.2f}"
        else:
            top_values = series.value_counts().nlargest(3)
            stats_str = "Top Values:\n" + "\n".join([f"  - '{str(idx)[:30]}': {val:,}" for idx, val in top_values.items()])

        # Create a container widget and a layout for it. This is the key to the fix.
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0) # Add a little vertical padding
        layout.setSpacing(0)

        label = QLabel(stats_str)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        label.setStyleSheet("background-color: transparent; border: none;")

        layout.addWidget(label)
        return container # Return the container, not just the label

    def filter_table(self, text):
        for i in range(self.column_table.rowCount()):
            item = self.column_table.item(i, 0)
            if item:
                match = text.lower() in item.text().lower()
                self.column_table.setRowHidden(i, not match)