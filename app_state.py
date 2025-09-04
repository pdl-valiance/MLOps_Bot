# app_state.py
import pandas as pd
from PyQt5.QtCore import QObject, pyqtSignal

class AppState(QObject):
    data_loaded = pyqtSignal(pd.DataFrame)
    data_cleaned = pyqtSignal(pd.DataFrame)
    data_aggregated = pyqtSignal(pd.DataFrame)

    def __init__(self):
        super().__init__()
        self.raw_df = None
        self.cleaned_df = None
        self.aggregated_df = None
        self.grouping_levels = []
        self.uploaded_dfs = {}

    def load_new_data(self, df: pd.DataFrame):
        self.raw_df = df
        self.cleaned_df = df.copy()
        self.aggregated_df = None
        self.uploaded_dfs = {} 
        print("AppState: New data loaded.")
        self.data_loaded.emit(self.cleaned_df)
    
    def update_cleaned_data(self, df: pd.DataFrame):
        self.cleaned_df = df
        self.aggregated_df = None
        print("AppState: Cleaned data updated.")
        self.data_cleaned.emit(self.cleaned_df)

    def update_aggregated_data(self, df: pd.DataFrame, grouping_levels: list):
        self.aggregated_df = df
        self.grouping_levels = grouping_levels
        print("AppState: Aggregated data created.")
        self.data_aggregated.emit(self.aggregated_df)