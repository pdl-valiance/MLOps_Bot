# ui/data_cleaner_tab.py

import pandas as pd
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QFrame, QLabel, 
                             QHBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView, 
                             QTabWidget, QListWidget, QListWidgetItem, QSpinBox, 
                             QAbstractItemView, QGridLayout)
from PyQt5.QtCore import Qt
from app_state import AppState

class DataCleanerTab(QWidget):
    def __init__(self, app_state: AppState):
        super().__init__()
        self.app_state = app_state
        self.duplicate_rows_df = None

        # --- Main Layout ---
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(25, 25, 25, 25)
        page_layout.setSpacing(15)
        page_layout.setAlignment(Qt.AlignTop)

        # --- Title and Description ---
        title = QLabel("Data Cleaning Operations")
        title.setObjectName("h1_label")
        description = QLabel("A suite of tools to handle common data quality issues like duplicates and missing values. Each operation modifies the dataset for subsequent steps.")
        description.setWordWrap(True)
        page_layout.addWidget(title)
        page_layout.addWidget(description)

        # --- Main Tab Widget for Organization ---
        main_tabs = QTabWidget()
        page_layout.addWidget(main_tabs, 1)

        duplicates_widget = self.create_duplicates_tab()
        missing_values_widget = self.create_missing_values_tab()

        main_tabs.addTab(duplicates_widget, "Handle Duplicates")
        main_tabs.addTab(missing_values_widget, "Handle Missing Values")

    def create_duplicates_tab(self):
        """Creates the widget for the 'Handle Duplicates' tab."""
        container = QWidget()
        content_layout = QHBoxLayout(container)
        content_layout.setContentsMargins(0, 15, 0, 0)

        # Left Pane: Column Selection & Actions
        selection_card = QFrame(); selection_card.setObjectName("card")
        selection_layout = QVBoxLayout(selection_card)
        selection_layout.addWidget(QLabel("<h4>Step 1: Select Columns for Duplicate Check</h4>"))
        self.dup_column_list = QListWidget()
        selection_layout.addWidget(self.dup_column_list)
        
        action_layout = QHBoxLayout()
        self.find_button = QPushButton("Find Duplicates"); self.find_button.clicked.connect(self.find_duplicates)
        self.drop_dup_button = QPushButton("Drop Duplicates"); self.drop_dup_button.clicked.connect(self.drop_duplicates)
        self.drop_dup_button.setEnabled(False)
        action_layout.addWidget(self.find_button); action_layout.addWidget(self.drop_dup_button)
        selection_layout.addLayout(action_layout)

        # Right Pane: Preview of Duplicates
        preview_card = QFrame(); preview_card.setObjectName("card")
        preview_layout = QVBoxLayout(preview_card)
        
        preview_title_layout = QHBoxLayout()
        preview_title_layout.addWidget(QLabel("<h4>Step 2: Preview Duplicates Found</h4>"))
        preview_title_layout.addStretch()
        self.dup_status_label = QLabel("Ready to check for duplicates.")
        self.dup_status_label.setObjectName("status_label")
        preview_title_layout.addWidget(self.dup_status_label)
        
        preview_layout.addLayout(preview_title_layout)
        self.dup_preview_table = QTableWidget()
        self.dup_preview_table.setEditTriggers(QTableWidget.NoEditTriggers)
        preview_layout.addWidget(self.dup_preview_table)

        content_layout.addWidget(selection_card, 1)
        content_layout.addWidget(preview_card, 3)
        return container

    def create_missing_values_tab(self):
        """Creates the widget for the 'Handle Missing Values' tab."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 15, 0, 0)
        layout.setSpacing(15)

        # Top section: Column-level operations
        column_ops_card = QFrame(); column_ops_card.setObjectName("card")
        column_ops_layout = QGridLayout(column_ops_card)
        column_ops_layout.setSpacing(15)
        
        # Column Stats Table
        table_container = QVBoxLayout()
        table_container.addWidget(QLabel("<h4>Column Null Value Analysis</h4>"))
        self.missing_val_table = QTableWidget()
        self.missing_val_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table_container.addWidget(self.missing_val_table)
        
        # Column Actions
        actions_container = QVBoxLayout()
        actions_container.setSpacing(10)
        actions_container.addWidget(QLabel("<h4>Column Actions</h4>"))
        
        self.drop_selected_cols_button = QPushButton("Drop Selected Columns")
        self.drop_selected_cols_button.clicked.connect(self.drop_selected_columns)
        
        threshold_layout = QHBoxLayout()
        self.drop_threshold_button = QPushButton("Drop if Null >")
        self.drop_threshold_button.clicked.connect(self.drop_columns_by_threshold)
        self.null_threshold_spinbox = QSpinBox()
        self.null_threshold_spinbox.setRange(0, 100)
        self.null_threshold_spinbox.setValue(50)
        self.null_threshold_spinbox.setSuffix(" %")
        threshold_layout.addWidget(self.drop_threshold_button)
        threshold_layout.addWidget(self.null_threshold_spinbox)
        
        actions_container.addWidget(self.drop_selected_cols_button)
        actions_container.addLayout(threshold_layout)
        actions_container.addStretch()
        
        column_ops_layout.addLayout(table_container, 0, 0, 1, 3) # Span 3 columns
        column_ops_layout.addLayout(actions_container, 0, 3, 1, 1) # Span 1 column

        # Bottom section: Row-level operations
        row_ops_card = QFrame(); row_ops_card.setObjectName("card")
        row_ops_layout = QHBoxLayout(row_ops_card)
        row_ops_layout.addWidget(QLabel("<h4>Row Actions</h4>"))
        row_ops_layout.addStretch()
        self.drop_na_rows_button = QPushButton("Drop All Rows With Any Nulls")
        self.drop_na_rows_button.clicked.connect(self.drop_rows_with_nulls)
        row_ops_layout.addWidget(self.drop_na_rows_button)

        layout.addWidget(column_ops_card)
        layout.addWidget(row_ops_card)
        layout.addStretch()
        return container

    def set_data(self, df: pd.DataFrame):
        """Called when data is loaded or updated, populates all sub-tabs."""
        self.populate_duplicates_tab(df)
        self.populate_missing_values_tab(df)

    # --- DUPLICATE HANDLERS ---
    def populate_duplicates_tab(self, df):
        self.dup_column_list.clear()
        self.dup_preview_table.clear(); self.dup_preview_table.setRowCount(0); self.dup_preview_table.setColumnCount(0)
        self.drop_dup_button.setEnabled(False)
        self.show_status("Ready to check for duplicates.", self.dup_status_label)

        if df is not None:
            for col in df.columns:
                item = QListWidgetItem(col)
                item.setCheckState(Qt.Checked)
                self.dup_column_list.addItem(item)

    def find_duplicates(self):
        df = self.app_state.cleaned_df
        if df is None: return self.show_status("No data loaded.", self.dup_status_label, True)
        selected_cols = [self.dup_column_list.item(i).text() for i in range(self.dup_column_list.count()) if self.dup_column_list.item(i).checkState() == Qt.Checked]
        if not selected_cols: return self.show_status("Please select at least one column.", self.dup_status_label, True)
        
        duplicates_mask = df.duplicated(subset=selected_cols, keep=False)
        self.duplicate_rows_df = df[duplicates_mask].sort_values(by=selected_cols)
        num_duplicates = len(self.duplicate_rows_df)
        
        self.show_status(f"Found {num_duplicates} duplicate rows.", self.dup_status_label, False)
        self.drop_dup_button.setEnabled(num_duplicates > 0)
        self.display_preview(self.duplicate_rows_df, self.dup_preview_table)

    def drop_duplicates(self):
        df = self.app_state.cleaned_df
        if df is None or self.duplicate_rows_df is None: return self.show_status("No duplicates to drop.", self.dup_status_label, True)
        selected_cols = [self.dup_column_list.item(i).text() for i in range(self.dup_column_list.count()) if self.dup_column_list.item(i).checkState() == Qt.Checked]
        
        rows_before = len(df)
        cleaned_df = df.drop_duplicates(subset=selected_cols, keep='first')
        rows_after = len(cleaned_df)
        
        self.app_state.update_cleaned_data(cleaned_df)
        self.show_status(f"Success! Removed {rows_before - rows_after} rows.", self.dup_status_label, False)
        self.find_duplicates()

    # --- MISSING VALUE HANDLERS ---
    def populate_missing_values_tab(self, df):
        self.missing_val_table.clear()
        self.missing_val_table.setRowCount(0)
        self.missing_val_table.setColumnCount(0)
        if df is None: return

        self.missing_val_table.setColumnCount(3)
        self.missing_val_table.setHorizontalHeaderLabels(["Column Name", "Null Count", "Null %"])
        self.missing_val_table.setRowCount(len(df.columns))

        for i, col in enumerate(df.columns):
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df) * 100) if len(df) > 0 else 0
            self.missing_val_table.setItem(i, 0, QTableWidgetItem(col))
            self.missing_val_table.setItem(i, 1, QTableWidgetItem(f"{null_count}"))
            self.missing_val_table.setItem(i, 2, QTableWidgetItem(f"{null_pct:.2f} %"))
        
        self.missing_val_table.resizeColumnsToContents()
        self.missing_val_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)

    def drop_selected_columns(self):
        selected_items = self.missing_val_table.selectedItems()
        if not selected_items: return
        
        cols_to_drop = list(set(self.missing_val_table.item(item.row(), 0).text() for item in selected_items))
        df = self.app_state.cleaned_df.drop(columns=cols_to_drop)
        self.app_state.update_cleaned_data(df)

    def drop_columns_by_threshold(self):
        df = self.app_state.cleaned_df
        threshold = self.null_threshold_spinbox.value()
        
        null_pct = (df.isnull().sum() / len(df)) * 100
        cols_to_drop = null_pct[null_pct > threshold].index.tolist()
        
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            self.app_state.update_cleaned_data(df)

    def drop_rows_with_nulls(self):
        df = self.app_state.cleaned_df.dropna()
        self.app_state.update_cleaned_data(df)

    # --- UTILITY METHODS ---
    def display_preview(self, df_preview: pd.DataFrame, table: QTableWidget):
        table.clear()
        if df_preview is None or df_preview.empty:
            table.setRowCount(0); table.setColumnCount(0); return

        table.setRowCount(len(df_preview)); table.setColumnCount(len(df_preview.columns))
        table.setHorizontalHeaderLabels(df_preview.columns)
        for r in range(len(df_preview)):
            for c, col_name in enumerate(df_preview.columns):
                table.setItem(r, c, QTableWidgetItem(str(df_preview.iat[r, c])))
        table.resizeColumnsToContents()
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)

    def show_status(self, message, label: QLabel, is_error=False):
        label.setText(message)
        label.setObjectName("status_error" if is_error else "status_success")
        label.style().polish(label)
