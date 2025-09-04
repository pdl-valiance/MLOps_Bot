# ui/multi_upload_tab.py

import os
import pandas as pd
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QFrame, QLabel, 
                             QFileDialog, QHBoxLayout, QListWidget, QListWidgetItem,
                             QComboBox, QTableWidget, QTableWidgetItem, QGridLayout,
                             QHeaderView, QAbstractItemView, QButtonGroup, QStackedWidget)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from app_state import AppState

class JoinKeyRow(QWidget):
    """A custom widget for a single row in the join key table."""
    def __init__(self, left_cols, right_cols, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self); layout.setContentsMargins(2, 2, 2, 2)
        self.left_key_combo = QComboBox(); self.left_key_combo.addItems(left_cols)
        self.right_key_combo = QComboBox(); self.right_key_combo.addItems(right_cols)
        self.remove_button = QPushButton(QIcon("assets/icons/trash.svg"), "")
        self.remove_button.setFixedSize(25, 25)
        layout.addWidget(self.left_key_combo); layout.addWidget(QLabel("=")); layout.addWidget(self.right_key_combo); layout.addWidget(self.remove_button)

class MultiUploadTab(QWidget):
    def __init__(self, app_state: AppState):
        super().__init__()
        self.app_state = app_state
        self.pipeline_steps = []
        self.final_df = None

        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(25, 25, 25, 25); page_layout.setSpacing(15)

        title = QLabel("Join Pipeline Builder"); title.setObjectName("h1_label")
        page_layout.addWidget(title)

        # --- Tab Buttons ---
        tabs_bar = QHBoxLayout()
        self.tabs_group = QButtonGroup(self)
        self.tabs_group.setExclusive(True)

        self.datasets_btn = QPushButton("1. Datasets"); self.datasets_btn.setCheckable(True)
        self.pipeline_btn = QPushButton("2. Pipeline Configuration"); self.pipeline_btn.setCheckable(True)
        self.preview_btn = QPushButton("3. Preview & Finalize"); self.preview_btn.setCheckable(True)
        
        tabs_bar.addWidget(self.datasets_btn); tabs_bar.addWidget(self.pipeline_btn); tabs_bar.addWidget(self.preview_btn)
        tabs_bar.addStretch()
        
        self.tabs_group.addButton(self.datasets_btn, 0)
        self.tabs_group.addButton(self.pipeline_btn, 1)
        self.tabs_group.addButton(self.preview_btn, 2)
        self.tabs_group.buttonClicked[int].connect(self.switch_view)
        
        page_layout.addLayout(tabs_bar)

        # --- Main Stacked Widget ---
        self.main_stack = QStackedWidget()
        self.main_stack.addWidget(self._create_datasets_page())
        self.main_stack.addWidget(self._create_pipeline_page())
        self.main_stack.addWidget(self._create_preview_page())
        page_layout.addWidget(self.main_stack, 1)

        # Initial State
        self.datasets_btn.setChecked(True)
        self.pipeline_btn.setEnabled(False)
        self.preview_btn.setEnabled(False)

    def _create_datasets_page(self):
        page = QFrame(); page.setObjectName("card")
        layout = QVBoxLayout(page)
        self.available_files_list = QListWidget(); self.available_files_list.itemDoubleClicked.connect(self.set_base_table)
        self.add_files_button = QPushButton("Add Datasets..."); self.add_files_button.clicked.connect(self.add_files)
        layout.addWidget(QLabel("<h4>Available Datasets</h4><i>Double-click a file to start a new pipeline.</i>"))
        layout.addWidget(self.available_files_list, 1)
        layout.addWidget(self.add_files_button)
        return page

    def _create_pipeline_page(self):
        page = QFrame(); page.setObjectName("card")
        layout = QHBoxLayout(page)
        
        self.pipeline_list = QListWidget(); self.pipeline_list.itemClicked.connect(self.display_step_config)
        self.pipeline_list.setMaximumWidth(300)

        self.step_config_area = QFrame(); self.step_config_area.setObjectName("card_light")
        self.step_config_layout = QVBoxLayout(self.step_config_area)
        self.step_config_area.setVisible(False)

        layout.addWidget(self.pipeline_list)
        layout.addWidget(self.step_config_area, 1)
        return page

    def _create_preview_page(self):
        page = QFrame(); page.setObjectName("card")
        layout = QVBoxLayout(page)
        self.preview_table = QTableWidget()
        layout.addWidget(self.preview_table, 1)

        action_layout = QHBoxLayout()
        self.status_label = QLabel("Ready.")
        self.finalize_button = QPushButton("Finalize & Use Data"); self.finalize_button.setObjectName("generateButton")
        self.finalize_button.clicked.connect(self.finalize_pipeline); self.finalize_button.setEnabled(False)
        action_layout.addWidget(self.status_label, 1)
        action_layout.addWidget(self.finalize_button)
        layout.addLayout(action_layout)
        return page
    
    def switch_view(self, index):
        self.main_stack.setCurrentIndex(index)

    def add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select CSVs", "", "CSV Files (*.csv)")
        for path in paths:
            filename = os.path.basename(path)
            if filename not in self.app_state.uploaded_dfs:
                df = pd.read_csv(path, low_memory=False)
                self.app_state.uploaded_dfs[filename] = df
                self.available_files_list.addItem(QListWidgetItem(filename))
        if not self.pipeline_steps and self.available_files_list.count() > 0:
            self.set_base_table(self.available_files_list.item(0))

    def set_base_table(self, item):
        self.pipeline_steps = [{'type': 'base', 'name': item.text()}]
        self.refresh_pipeline_list()
        self.run_pipeline(preview=True)
        self.pipeline_btn.setEnabled(True)
        self.preview_btn.setEnabled(True)
        self.switch_view(1) # Switch to pipeline config view

    def refresh_pipeline_list(self):
        self.pipeline_list.clear()
        for i, step in enumerate(self.pipeline_steps):
            item_widget = QWidget()
            item_layout = QHBoxLayout(item_widget); item_layout.setContentsMargins(5, 5, 5, 5)
            label_text = f"<b>Base:</b> {step['name']}" if step['type'] == 'base' else f"<b>Join {i}:</b> with {step.get('right_table', '...')}"
            item_layout.addWidget(QLabel(label_text), 1)
            if step['type'] == 'join':
                delete_btn = QPushButton("🗑️"); delete_btn.setFixedSize(25, 25)
                delete_btn.clicked.connect(lambda ch, index=i: self.delete_step(index))
                item_layout.addWidget(delete_btn)
            list_item = QListWidgetItem(self.pipeline_list)
            list_item.setSizeHint(item_widget.sizeHint())
            self.pipeline_list.addItem(list_item); self.pipeline_list.setItemWidget(list_item, item_widget)
        self.pipeline_list.addItem(QListWidgetItem("➕ Add Join Step..."))

    def delete_step(self, index):
        if 0 < index < len(self.pipeline_steps):
            del self.pipeline_steps[index]
            self.refresh_pipeline_list()
            self.step_config_area.setVisible(False)
            self.run_pipeline(preview=True)

    def display_step_config(self, item):
        row = self.pipeline_list.row(item)
        if row >= len(self.pipeline_steps):
            if not self.pipeline_steps: return
            self.pipeline_steps.append({'type': 'join', 'keys': []}); self.refresh_pipeline_list()
        self.generate_config_ui(row)

    def generate_config_ui(self, index):
        # --- FIX: Robustly clear the layout of all items (widgets and nested layouts) ---
        while self.step_config_layout.count():
            item = self.step_config_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                layout = item.layout()
                if layout is not None:
                    # Recursively clear nested layout
                    while layout.count():
                        nested_item = layout.takeAt(0)
                        nested_widget = nested_item.widget()
                        if nested_widget is not None:
                            nested_widget.deleteLater()
        
        if index == 0 or index >= len(self.pipeline_steps):
            self.step_config_area.setVisible(False)
            return
        
        self.step_config_area.setVisible(True)
        step = self.pipeline_steps[index]
        self.step_config_layout.addWidget(QLabel(f"<h4>Configure Join Step {index}</h4>"))
        
        right_table_combo = QComboBox()
        right_table_combo.addItems(self.app_state.uploaded_dfs.keys())
        join_type_combo = QComboBox()
        join_type_combo.addItems(["inner", "left", "right", "outer"])
        
        self.join_keys_table = QTableWidget()
        self.join_keys_table.setColumnCount(1)
        self.join_keys_table.setHorizontalHeaderLabels(["Left Key = Right Key"])
        self.join_keys_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        add_key_btn = QPushButton("+")
        suggest_keys_btn = QPushButton("Suggest Keys")
        
        self.step_config_layout.addWidget(QLabel("Join with Table:"))
        self.step_config_layout.addWidget(right_table_combo)
        self.step_config_layout.addWidget(QLabel("Join Type:"))
        self.step_config_layout.addWidget(join_type_combo)
        self.step_config_layout.addWidget(QLabel("<b>Join On:</b>"))
        self.step_config_layout.addWidget(self.join_keys_table, 1)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(suggest_keys_btn)
        btn_layout.addWidget(add_key_btn)
        self.step_config_layout.addLayout(btn_layout)
        
        apply_btn = QPushButton("Apply Step")
        apply_btn.setObjectName("generateButton")
        self.step_config_layout.addWidget(apply_btn, 0, Qt.AlignRight)

        left_df_preview = self.run_pipeline(preview=True, up_to_step=index)
        if left_df_preview is None:
            self.status_label.setText(f"Cannot configure: previous step failed.")
            return

        def on_right_table_change(table_name):
            self.join_keys_table.setRowCount(0)
        right_table_combo.currentTextChanged.connect(on_right_table_change)

        def add_key_row_ui(left_key=None, right_key=None):
            right_df_name = right_table_combo.currentText()
            if not right_df_name: return
            right_df_cols = self.app_state.uploaded_dfs[right_df_name].columns
            row_count = self.join_keys_table.rowCount()
            self.join_keys_table.insertRow(row_count)
            row_widget = JoinKeyRow(left_df_preview.columns, right_df_cols)
            if left_key: row_widget.left_key_combo.setCurrentText(left_key)
            if right_key: row_widget.right_key_combo.setCurrentText(right_key)
            self.join_keys_table.setCellWidget(row_count, 0, row_widget)
            self.join_keys_table.resizeRowToContents(row_count)
            def remove_this_row():
                for r in range(self.join_keys_table.rowCount()):
                    if self.join_keys_table.cellWidget(r, 0) == row_widget:
                        self.join_keys_table.removeRow(r)
                        break
            row_widget.remove_button.clicked.connect(remove_this_row)

        add_key_btn.clicked.connect(add_key_row_ui)
        suggest_keys_btn.clicked.connect(lambda: self._suggest_keys_ui(left_df_preview, self.app_state.uploaded_dfs[right_table_combo.currentText()], add_key_row_ui))
        apply_btn.clicked.connect(lambda: self._apply_step_config(index, right_table_combo.currentText(), join_type_combo.currentText()))
        
        if step.get('right_table'): right_table_combo.setCurrentText(step['right_table'])
        if step.get('how'): join_type_combo.setCurrentText(step['how'])
        for l_key, r_key in step.get('keys', []):
            add_key_row_ui(l_key, r_key)

    def _suggest_keys_ui(self, left_df, right_df, add_row_func):
        common_cols = set(left_df.columns).intersection(set(right_df.columns))
        self.join_keys_table.setRowCount(0)
        for col in common_cols:
            add_row_func(left_key=col, right_key=col)

    def _apply_step_config(self, index, right_table, how):
        step = self.pipeline_steps[index]
        step['right_table'], step['how'] = right_table, how
        step['keys'] = []
        for r in range(self.join_keys_table.rowCount()):
            widget = self.join_keys_table.cellWidget(r, 0)
            step['keys'].append((widget.left_key_combo.currentText(), widget.right_key_combo.currentText()))
        
        self.refresh_pipeline_list()
        self.run_pipeline(preview=True)
        self.switch_view(2) # Switch to preview after applying

    def run_pipeline(self, preview=False, up_to_step=None):
        if not self.pipeline_steps: return None
        limit = 100 if preview else None
        end_step = up_to_step if up_to_step is not None else len(self.pipeline_steps)

        try:
            current_df = self.app_state.uploaded_dfs[self.pipeline_steps[0]['name']].copy()
            if limit: current_df = current_df.head(limit)

            for i in range(1, end_step):
                step = self.pipeline_steps[i]
                right_df_name, how, keys = step.get('right_table'), step.get('how'), step.get('keys')
                if not all([right_df_name, how, keys]): raise ValueError(f"Step {i} not fully configured.")
                
                right_df = self.app_state.uploaded_dfs[right_df_name].copy()
                if limit: right_df = right_df.head(limit)
                
                left_on, right_on = zip(*keys)
                current_df = pd.merge(current_df, right_df, how=how, left_on=list(left_on), right_on=list(right_on), suffixes=(f'_L{i-1}', f'_R{i}'))
            
            self.final_df = current_df
            if preview:
                self.display_in_table(current_df)
                self.status_label.setText(f"Preview ready ({len(current_df)} rows from sample).")
                self.finalize_button.setEnabled(True)
            return current_df
        except Exception as e:
            self.status_label.setText(f"Pipeline Error: {e}"); self.finalize_button.setEnabled(False); return None

    def finalize_pipeline(self):
        final_df = self.run_pipeline(preview=False)
        if final_df is not None:
            self.app_state.load_new_data(final_df)
            self.status_label.setText("Success! Pipeline data loaded for analysis.")
        else:
            self.status_label.setText("Pipeline failed. Cannot finalize.")

    def display_in_table(self, df):
        self.preview_table.clear(); self.preview_table.setRowCount(len(df)); self.preview_table.setColumnCount(len(df.columns))
        self.preview_table.setHorizontalHeaderLabels(df.columns)
        for r, row in enumerate(df.itertuples(index=False)):
            for c, value in enumerate(row):
                self.preview_table.setItem(r, c, QTableWidgetItem(str(value)))
        self.preview_table.resizeColumnsToContents()