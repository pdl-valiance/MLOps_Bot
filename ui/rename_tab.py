# ui/rename_tab.py

import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QFrame, QLabel,
                             QTableWidget, QTableWidgetItem,
                             QHeaderView, QLineEdit, QComboBox, QHBoxLayout)
from PyQt5.QtCore import Qt
from app_state import AppState

class RenameTab(QWidget):
    def __init__(self, app_state: AppState):
        super().__init__()
        self.app_state = app_state
        self.df = None
        self.input_widgets = []

        # --- Constants ---
        self.TYPE_OPTIONS = ['string', 'int', 'float', 'bool', 'datetime', 'category']

        # --- Main Layout ---
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(25, 25, 25, 25)
        page_layout.setSpacing(15)
        page_layout.setAlignment(Qt.AlignTop)

        # Title and description
        title = QLabel("Column Renaming & Data Type Conversion")
        title.setObjectName("h1_label")
        description = QLabel("Clean column names and correct their data types. Use 'category' for text columns with few unique values to improve performance.")
        description.setWordWrap(True)
        page_layout.addWidget(title)
        page_layout.addWidget(description)
        
        # --- Filter Bar ---
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter Columns:"))
        self.filter_edit = QLineEdit()
        self.filter_edit.setObjectName("filter_bar")
        self.filter_edit.setPlaceholderText("Enter column name to filter...")
        self.filter_edit.textChanged.connect(self.filter_table)
        filter_layout.addWidget(self.filter_edit)
        page_layout.addLayout(filter_layout)
        
        # --- CORRECTED: Action & Status Area is now at the top ---
        action_frame = QFrame()
        action_layout = QHBoxLayout(action_frame)
        action_layout.setContentsMargins(0, 0, 0, 0)
        self.status_label = QLabel("Ready to apply changes.")
        self.status_label.setObjectName("status_label")
        self.status_label.setWordWrap(True)
        self.apply_button = QPushButton("Apply Changes")
        self.apply_button.clicked.connect(self.apply_changes)
        
        action_layout.addWidget(self.status_label)
        action_layout.addStretch()
        action_layout.addWidget(self.apply_button)
        page_layout.addWidget(action_frame)
        
        # --- Table Card ---
        table_card = QFrame()
        table_card.setObjectName("card")
        card_layout = QVBoxLayout(table_card)
        self.table = QTableWidget()
        card_layout.addWidget(self.table)
        page_layout.addWidget(table_card, 1) # Give stretch factor to the card

    def set_data(self, df: pd.DataFrame):
        self.df = df
        if df is not None:
            self.populate_table()

    def populate_table(self):
        df = self.app_state.cleaned_df
        if df is None: return

        self.input_widgets = []
        self.table.clear()
        self.table.setRowCount(len(df.columns))
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Original Name", "New Name", "Data Type"])
        
        for row, col in enumerate(df.columns):
            original_name_item = QTableWidgetItem(col)
            original_name_item.setFlags(original_name_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, original_name_item)

            suggested_name = col.strip().lower().replace(" ", "_").replace("(", "").replace(")", "")
            new_name_edit = QLineEdit(suggested_name)
            self.table.setCellWidget(row, 1, new_name_edit)

            type_combo = QComboBox()
            type_combo.addItems(self.TYPE_OPTIONS)
            current_type = self.get_mapped_type(df[col].dtype)
            type_combo.setCurrentText(current_type)
            self.table.setCellWidget(row, 2, type_combo)
            
            self.input_widgets.append({'original': col, 'name_edit': new_name_edit, 'type_combo': type_combo})

        self.table.resizeRowsToContents()
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)

    def filter_table(self, text):
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item:
                match = text.lower() in item.text().lower()
                self.table.setRowHidden(row, not match)
                
    def apply_changes(self):
        if not self.input_widgets or self.app_state.cleaned_df is None:
            self.show_status("No data loaded to process.", is_error=True)
            return

        temp_df = self.app_state.cleaned_df.copy()
        
        new_names_map = {item['original']: item['name_edit'].text().strip() for item in self.input_widgets}
        if len(new_names_map.values()) != len(set(new_names_map.values())):
            self.show_status("Error: Duplicate new column names detected.", is_error=True)
            return
            
        temp_df.rename(columns=new_names_map, inplace=True)
        
        errors = []
        for item in self.input_widgets:
            new_col = new_names_map[item['original']]
            target_type = item['type_combo'].currentText()
            try:
                if self.get_mapped_type(temp_df[new_col].dtype) == target_type: continue
                if target_type == 'string': temp_df[new_col] = temp_df[new_col].astype(str)
                elif target_type == 'int': temp_df[new_col] = pd.to_numeric(temp_df[new_col], errors='coerce').astype('Int64')
                elif target_type == 'float': temp_df[new_col] = pd.to_numeric(temp_df[new_col], errors='coerce').astype(float)
                elif target_type == 'bool': temp_df[new_col] = temp_df[new_col].astype(str).str.lower().map({'true':True, '1':True, 'yes':True, 't':True,'false':False, '0':False, 'no':False, 'f':False}).astype('boolean')
                elif target_type == 'datetime': temp_df[new_col] = pd.to_datetime(temp_df[new_col], errors='coerce')
                elif target_type == 'category': temp_df[new_col] = temp_df[new_col].astype('category')
            except Exception as e:
                errors.append(f"Could not convert '{new_col}' to {target_type}")

        if errors:
            self.show_status("Completed with errors: " + ", ".join(errors), is_error=True)
        else:
            self.show_status("Success! All changes have been applied.", is_error=False)

        self.app_state.update_cleaned_data(temp_df)

    def get_mapped_type(self, dtype):
        if pd.api.types.is_datetime64_any_dtype(dtype): return 'datetime'
        if pd.api.types.is_integer_dtype(dtype): return 'int'
        if pd.api.types.is_float_dtype(dtype): return 'float'
        if pd.api.types.is_bool_dtype(dtype): return 'bool'
        if pd.api.types.is_categorical_dtype(dtype): return 'category'
        return 'string'

    def show_status(self, message, is_error=False):
        self.status_label.setText(message)
        self.status_label.setObjectName("status_error" if is_error else "status_success")
        self.status_label.style().polish(self.status_label)