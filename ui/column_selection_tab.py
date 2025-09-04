# ui/column_selection_tab.py

import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QFrame, QLabel, 
                             QGridLayout, QComboBox, QHBoxLayout, QListWidget, QListWidgetItem,
                             QDateEdit)
from PyQt5.QtCore import Qt, QDate
from app_state import AppState

class ColumnSelectionTab(QWidget):
    def __init__(self, app_state: AppState):
        super().__init__()
        self.app_state = app_state
        
        # Constants
        self.DATE_FORMATS = ['%Y-%m-%d %H:%M:%S', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y', '%Y/%m/%d', '%Y-%m', '%m-%Y']
        self.PRIORITY_KEYWORDS = ['qty','price','sales', 'value', 'revenue', 'amount', 'unit', 'grand_total', 'price']

        # --- Main Layout ---
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(25, 25, 25, 25)
        page_layout.setSpacing(20)
        page_layout.setAlignment(Qt.AlignTop)

        title = QLabel("Column Configuration & Pre-processing")
        title.setObjectName("h1_label")
        page_layout.addWidget(title)
        
        # --- Step 1: Column Selection ---
        select_card = QFrame(); select_card.setObjectName("card")
        select_layout = QVBoxLayout(select_card)
        select_layout.addWidget(QLabel("<h3>Step 1: Select Columns to Keep</h3>"))
        
        list_layout = QHBoxLayout()
        
        available_layout = QVBoxLayout()
        available_layout.addWidget(QLabel("<b>Unused Columns</b> (Drag here to exclude)"))
        self.available_list = QListWidget()
        self.available_list.setDragDropMode(QListWidget.DragDrop)
        self.available_list.setDefaultDropAction(Qt.MoveAction)
        available_layout.addWidget(self.available_list)
        
        selected_layout = QVBoxLayout()
        selected_layout.addWidget(QLabel("<b>Columns to Keep for Analysis</b>"))
        self.selected_list = QListWidget()
        self.selected_list.setDragDropMode(QListWidget.DragDrop)
        self.selected_list.setDefaultDropAction(Qt.MoveAction)
        
        self.selected_list.model().rowsMoved.connect(self.update_dropdowns)
        self.available_list.model().rowsMoved.connect(self.update_dropdowns)
        
        selected_layout.addWidget(self.selected_list)
        
        list_layout.addLayout(available_layout)
        list_layout.addLayout(selected_layout)
        select_layout.addLayout(list_layout)
        page_layout.addWidget(select_card)

        # --- Step 2 & 3 in a single row ---
        bottom_layout = QHBoxLayout()
        page_layout.addLayout(bottom_layout)

        # --- Step 2: Date/Target Config & Filter ---
        config_card = QFrame(); config_card.setObjectName("card")
        config_layout = QVBoxLayout(config_card)
        config_layout.addWidget(QLabel("<h3>Step 2: Configure Date & Filters</h3>"))
        
        self.date_col_combo = QComboBox(); self.date_format_combo = QComboBox()
        self.date_format_combo.addItems(self.DATE_FORMATS)
        
        # --- THIS IS THE FIX: Connect signals to update the date range interactively ---
        self.date_col_combo.currentIndexChanged.connect(self.update_date_filter_range)
        self.date_format_combo.currentIndexChanged.connect(self.update_date_filter_range)
        
        form_layout = QGridLayout(); form_layout.setSpacing(15)
        form_layout.addWidget(QLabel("Date Column:"), 0, 0); form_layout.addWidget(self.date_col_combo, 0, 1)
        form_layout.addWidget(QLabel("Date Format:"), 1, 0); form_layout.addWidget(self.date_format_combo, 1, 1)
        
        self.start_date_edit = QDateEdit(); self.start_date_edit.setCalendarPopup(True)
        self.end_date_edit = QDateEdit(); self.end_date_edit.setCalendarPopup(True)
        form_layout.addWidget(QLabel("Filter Start Date:"), 2, 0); form_layout.addWidget(self.start_date_edit, 2, 1)
        form_layout.addWidget(QLabel("Filter End Date:"), 3, 0); form_layout.addWidget(self.end_date_edit, 3, 1)

        config_layout.addLayout(form_layout)
        config_layout.addStretch()
        bottom_layout.addWidget(config_card, 1)

        # --- Step 3: Target Column ---
        target_card = QFrame(); target_card.setObjectName("card")
        target_layout = QVBoxLayout(target_card)
        target_layout.addWidget(QLabel("<h3>Step 3: Select Target</h3>"))
        target_layout.addWidget(QLabel("Target Column (for analysis):"))
        self.target_col_combo = QComboBox()
        target_layout.addWidget(self.target_col_combo)
        target_layout.addStretch()
        bottom_layout.addWidget(target_card, 1)

        # --- Action Bar ---
        action_layout = QHBoxLayout()
        self.status_label = QLabel("Ready."); self.status_label.setObjectName("status_label")
        self.apply_button = QPushButton("Apply Configuration"); self.apply_button.clicked.connect(self.apply_changes)
        action_layout.addWidget(self.status_label); action_layout.addStretch(); action_layout.addWidget(self.apply_button)
        page_layout.addLayout(action_layout)

    def set_data(self, df: pd.DataFrame):
        self.available_list.clear(); self.selected_list.clear()
        if df is None:
            self.date_col_combo.clear(); self.target_col_combo.clear(); return

        self.selected_list.addItems(df.columns)
        self.update_dropdowns()

    def update_dropdowns(self):
        current_cols = [self.selected_list.item(i).text() for i in range(self.selected_list.count())]
        if not current_cols or self.app_state.cleaned_df is None:
            self.date_col_combo.clear(); self.target_col_combo.clear(); return
            
        df_subset = self.app_state.cleaned_df[current_cols]

        potential_date_cols = [col for col in current_cols if df_subset[col].dtype == 'object' or pd.api.types.is_datetime64_any_dtype(df_subset[col])]
        numeric_cols = df_subset.select_dtypes(include=np.number).columns.tolist()
        priority_targets = [col for col in numeric_cols if any(k in col.lower() for k in self.PRIORITY_KEYWORDS)]
        other_targets = [col for col in numeric_cols if col not in priority_targets]
        
        self.date_col_combo.blockSignals(True)
        self.date_col_combo.clear(); self.date_col_combo.addItems(potential_date_cols)
        self.date_col_combo.blockSignals(False)
        
        self.target_col_combo.clear(); self.target_col_combo.addItems(priority_targets + other_targets)
        
        self.update_date_filter_range()

    def update_date_filter_range(self):
        date_col = self.date_col_combo.currentText()
        date_format = self.date_format_combo.currentText()
        
        if not date_col or not date_format or self.app_state.cleaned_df is None: return

        try:
            date_series = pd.to_datetime(self.app_state.cleaned_df[date_col], format=date_format, errors='coerce')
            date_series.dropna(inplace=True)

            if not date_series.empty:
                min_date, max_date = date_series.min(), date_series.max()
                self.start_date_edit.setDate(QDate(min_date.year, min_date.month, min_date.day))
                self.end_date_edit.setDate(QDate(max_date.year, max_date.month, max_date.day))
        except Exception as e:
            print(f"Could not auto-set date filter range: {e}")

    def apply_changes(self):
        if self.app_state.cleaned_df is None:
            self.show_status("Error: No data loaded.", is_error=True); return

        cols_to_keep = [self.selected_list.item(i).text() for i in range(self.selected_list.count())]
        if not cols_to_keep:
            self.show_status("Error: You must select at least one column to keep.", is_error=True); return
        
        temp_df = self.app_state.cleaned_df[cols_to_keep].copy()
        
        date_col = self.date_col_combo.currentText()
        date_format = self.date_format_combo.currentText()
        if not date_col:
            self.show_status("Error: Please select a Date column from your kept columns.", is_error=True); return

        try:
            temp_df[date_col] = pd.to_datetime(temp_df[date_col], format=date_format, errors='coerce')
            if temp_df[date_col].isnull().all():
                self.show_status(f"Error: All values in '{date_col}' became null. Check format.", is_error=True); return
            
            start_date = self.start_date_edit.date().toPyDate()
            end_date = self.end_date_edit.date().toPyDate()
            temp_df = temp_df[(temp_df[date_col].dt.date >= start_date) & (temp_df[date_col].dt.date <= end_date)]
            
            self.app_state.update_cleaned_data(temp_df)
            self.show_status(f"Success! Configuration applied. Data has {temp_df.shape[0]} rows and {temp_df.shape[1]} columns.", is_error=False)
            
        except Exception as e:
            self.show_status(f"Processing Error: {e}", is_error=True)

    def show_status(self, message, is_error=False):
        self.status_label.setText(message)
        self.status_label.setObjectName("status_error" if is_error else "status_success")
        self.status_label.style().polish(self.status_label)
