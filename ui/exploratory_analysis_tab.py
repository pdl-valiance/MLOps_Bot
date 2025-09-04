# ui/exploratory_analysis_tab.py

import pandas as pd
import io
from contextlib import redirect_stdout
import numpy as np
from scipy.stats import zscore
import shutil, os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QFrame, QLabel, QComboBox,
                             QHBoxLayout, QSizePolicy, QScrollArea, QTabWidget, QTableWidget,
                             QTableWidgetItem, QHeaderView, QSpinBox, QTextEdit, QProgressBar,
                             QGridLayout, QListWidget, QListWidgetItem, QLineEdit,QCheckBox,QStackedWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QDoubleValidator


# Matplotlib and Statsmodels for plotting
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# AutoGluon for AutoML
from autogluon.tabular import TabularPredictor

from app_state import AppState

# --- Global Configuration for Table Row Height ---
DEFAULT_TABLE_ROW_HEIGHT = 150 # Increased default row height for better readability and to accommodate wrapping

# --- Reusable Matplotlib Widget ---
class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#ffffff')
        super().__init__(self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

# --- Base Class for all EDA Pages ---
class BaseEdaPage(QWidget):
    def __init__(self, app_state: AppState):
        super().__init__()
        self.app_state = app_state; self.df = None; self.date_col = None
        self.numeric_cols = []; self.categorical_cols = []; self.grouping_cols = []
        self.PRIORITY_KEYWORDS = ['qty','price','sales', 'value', 'revenue', 'amount', 'unit', 'grand_total', 'price']
        self.numeric_cols = []; self.categorical_cols = []; self.grouping_cols = []
        
        # Main page layout with consistent margins and spacing
        self.page_layout = QVBoxLayout(self)
        self.page_layout.setContentsMargins(25, 25, 25, 25) # Consistent padding
        self.page_layout.setSpacing(20) # Increased spacing between major sections
        self.page_layout.setAlignment(Qt.AlignTop)

        # Title Label
        self.title_label = QLabel("Analysis Page")
        self.title_label.setObjectName("h1_label")
        self.page_layout.addWidget(self.title_label)

        # Controls Card - consistent styling for input area
        controls_card = QFrame(); controls_card.setObjectName("card")
        self.controls_layout = QHBoxLayout(controls_card)
        self.controls_layout.setContentsMargins(15, 15, 15, 15) # Inner padding for controls card
        self.controls_layout.setSpacing(10) # Spacing between controls
        self.controls_layout.setAlignment(Qt.AlignLeft) # Align controls to the left
        self.page_layout.addWidget(controls_card)
        
        # Plot Area - wrapped in a QScrollArea for consistent scrolling behavior
        plot_area = QScrollArea()
        plot_area.setWidgetResizable(True)
        # Styled to blend with the background, no visible border
        plot_area.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        
        plot_container_widget = QWidget()
        plot_container_widget.setStyleSheet("background-color: transparent;") # Ensure inner widget is transparent
        
        self.plot_layout = QVBoxLayout(plot_container_widget)
        self.plot_layout.setContentsMargins(0, 0, 0, 0) # No extra margins inside the scrollable container
        self.plot_layout.setAlignment(Qt.AlignTop)
        self.plot_layout.setSpacing(25) # Spacing between plots/cards
        
        plot_area.setWidget(plot_container_widget)
        self.page_layout.addWidget(plot_area, 1) # Give stretch factor to the scroll area
    def create_table_widget(self, df, stretch_columns=None, max_visible_rows=15):
        """
        Creates a standardized, well-behaved QTableWidget.
        This new version fixes column stretching and height issues.
        """
        table = QTableWidget()
        table.setRowCount(len(df))
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels(df.columns)
        
        # --- Standard Settings for Better UI ---
        table.setSortingEnabled(True)
        table.setWordWrap(True)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setAlternatingRowColors(True)

        # --- Fill Data ---
        for r, row in enumerate(df.itertuples(index=False)):
            for c, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                if isinstance(value, (int, float, np.number)):
                    # Align numbers to the right
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                else:
                    item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                table.setItem(r, c, item)

        # --- Sizing Policies (The Key Fix) ---
        header = table.horizontalHeader()
        # Stretch all columns to fill available horizontal space, preventing text cutoff
        for i in range(header.count()):
            header.setSectionResizeMode(i, QHeaderView.Stretch)
        
        # Make rows tall enough for their content
        table.resizeRowsToContents()

        # --- Intelligent Height Limiting ---
        if table.rowCount() > max_visible_rows:
            header_height = table.horizontalHeader().height()
            # Calculate height for the max number of visible rows
            total_rows_height = sum(table.rowHeight(i) for i in range(max_visible_rows))
            # Set a fixed height to enable the table's own scrollbar
            table.setFixedHeight(header_height + total_rows_height + 5)
        else:
            # If the table is small, let it take its natural height
            total_rows_height = sum(table.rowHeight(i) for i in range(table.rowCount()))
            table.setFixedHeight(table.horizontalHeader().height() + total_rows_height + 5)

        return table

    def set_data(self, df: pd.DataFrame):
        self.df = df
        if df is None: return
        self.date_col = next((c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])), None)
        self.numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.grouping_cols = self.app_state.grouping_levels
        self.update_controls()
    
    def update_controls(self): pass
    def run_analysis(self): self.clear_plots()
    def clear_plots(self):
        while self.plot_layout.count():
            child = self.plot_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
    def create_separator(self):
        sep = QFrame(); sep.setFrameShape(QFrame.HLine); sep.setObjectName("plot_separator"); return sep

# --- Worker thread for running AutoML in the background ---
class AutoMLWorker(QThread):
    finished = pyqtSignal(object, str)
    error = pyqtSignal(str)

    def __init__(self, df, target, time_limit, segment_name):
        super().__init__()
        self.df = df
        self.target = target
        self.time_limit = time_limit
        self.segment_name = segment_name

    def run(self):
        try:
            path = f'./autogluon_models/model_{self.segment_name}'
            if os.path.exists(path): shutil.rmtree(path)
            
            log_stream = io.StringIO()
            with redirect_stdout(log_stream):
                predictor = TabularPredictor(label=self.target, path=path).fit(
                    self.df, time_limit=self.time_limit
                )
                leaderboard = predictor.leaderboard(self.df, silent=True)
            
            logs = log_stream.getvalue()
            self.finished.emit(leaderboard, logs)
        except Exception as e:
            self.error.emit(f"AutoML Training Error: {e}")


# --- Analysis Page 1: Time Series & Core Insights ---
class TimeSeriesPage(BaseEdaPage):
    def __init__(self, app_state):
        super().__init__(app_state)
        self.title_label.setText("Time Series Analysis")
        
        # Controls layout within the controls_card
        self.controls_layout.addWidget(QLabel("Segment:"))
        self.group_combo = QComboBox()
        self.group_combo.setMinimumWidth(150)
        self.controls_layout.addWidget(self.group_combo, 1)
        
        self.controls_layout.addWidget(QLabel("Target:"))
        self.target_combo = QComboBox()
        self.target_combo.setMinimumWidth(150)
        self.controls_layout.addWidget(self.target_combo, 1)
        
        self.run_button = QPushButton("Generate Analysis")
        self.run_button.clicked.connect(self.run_analysis)
        self.controls_layout.addWidget(self.run_button)
        self.controls_layout.addStretch()

    def update_controls(self):
        self.target_combo.clear()
        self.target_combo.addItems(self.numeric_cols)
        self.group_combo.clear()
        self.group_combo.addItem("All Segments")
        if self.grouping_cols:
            unique_groups = self.df[self.grouping_cols].drop_duplicates().values
            self.group_combo.addItems([" | ".join(map(str, c)) for c in unique_groups])
        else:
            self.group_combo.addItem("Overall")

    def _calculate_decomposition_stats(self, time_series):
        """Calculates trend and seasonality strength."""
        if len(time_series) < 15:
            return 0, 0
        try:
            period = min(12, len(time_series) // 2)
            if period < 2: return 0, 0
            
            result = seasonal_decompose(time_series, model='additive', period=period, extrapolate_trend='freq')
            
            # Replace NaN in trend and seasonal with 0 for variance calculation
            result.trend.fillna(0, inplace=True)
            result.seasonal.fillna(0, inplace=True)
            
            detrended = time_series - result.trend
            deseasonalized = time_series - result.seasonal
            
            trend_strength = max(0, 1 - np.var(result.resid.dropna()) / np.var(deseasonalized.dropna()))
            seasonal_strength = max(0, 1 - np.var(result.resid.dropna()) / np.var(detrended.dropna()))
            
            return trend_strength, seasonal_strength
        except:
            return 0, 0

    def _calculate_anomaly_count(self, time_series):
        """Calculates the number of anomalies."""
        ts_clean = time_series.dropna()
        if len(ts_clean) < 2:
            return 0
        try:
            z_scores = np.abs(zscore(ts_clean))
            return np.sum(z_scores > 3)
        except:
            return 0

    def run_analysis(self):
        super().run_analysis()
        target_col = self.target_combo.currentText()
        if self.df is None or self.date_col is None or not target_col:
            error_label = QLabel("No valid data or target selected.")
            error_label.setStyleSheet("font-size: 14px; color: #666; padding: 20px;")
            self.plot_layout.addWidget(error_label)
            return

        # Main Tab container for Summary and Details
        results_tabs = QTabWidget()
        results_tabs.setMinimumHeight(600)  # Ensure minimum height
        self.plot_layout.addWidget(results_tabs)

        # Get all segments to analyze
        selected_text = self.group_combo.currentText()
        if selected_text == "All Segments":
            groups = [("Overall", self.df)] if not self.grouping_cols else \
                     [( " | ".join(map(str,row)), self.df[(self.df[self.grouping_cols] == row).all(axis=1)]) 
                      for _, row in self.df[self.grouping_cols].drop_duplicates().iterrows()]
        else:
            segment_df = self.df
            if selected_text != "Overall":
                selected_values = selected_text.split(" | ")
                mask = (self.df[self.grouping_cols].astype(str).values == selected_values).all(axis=1)
                segment_df = self.df[mask]
            groups = [(selected_text, segment_df)]

        summary_data = []
        for name, group_df in groups:
            ts = group_df.set_index(self.date_col)[target_col].sort_index()
            trend, season = self._calculate_decomposition_stats(ts)
            anomalies = self._calculate_anomaly_count(ts)
            summary_data.append({
                "Segment": name, "Trend Strength": trend,
                "Seasonality Strength": season, "Anomalies Found": anomalies
            })
        
        summary_df = pd.DataFrame(summary_data)

        summary_widget = self._create_summary_widget(summary_df)
        results_tabs.addTab(summary_widget, "📊 Summary Dashboard")
        
        detailed_widget = QScrollArea()
        detailed_widget.setWidgetResizable(True)
        detailed_widget.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        detailed_widget.setMinimumHeight(500)  # Minimum height for scroll area
        
        container = QWidget()
        detailed_layout = QVBoxLayout(container)
        
        # Set proper layout properties to prevent squeezing
        detailed_layout.setAlignment(Qt.AlignTop)
        detailed_layout.setSpacing(20)
        detailed_layout.setContentsMargins(10, 10, 10, 10)
        
        detailed_widget.setWidget(container)

        for name, group_df in groups:
            segment_card = self.perform_segment_analysis(group_df, name, target_col)
            if segment_card:
                segment_card.setMinimumHeight(800)  # Ensure minimum height for each segment card
                detailed_layout.addWidget(segment_card)
        
        # Add stretch at the end to prevent compression
        detailed_layout.addStretch()
        
        results_tabs.addTab(detailed_widget, "🔍 Detailed Analysis")

    def _create_summary_widget(self, summary_df: pd.DataFrame):
        widget = QScrollArea()
        widget.setWidgetResizable(True)
        widget.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        widget.setMinimumHeight(500)
        
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(20)
        layout.setContentsMargins(10, 10, 10, 10)
        widget.setWidget(container)

        # KPI Cards with fixed height
        kpi_layout = QGridLayout()
        total_segments = len(summary_df) if not summary_df.empty else 1
        strong_trend_count = (summary_df["Trend Strength"] > 0.6).sum()
        strong_season_count = (summary_df["Seasonality Strength"] > 0.6).sum()
        anomaly_count = (summary_df["Anomalies Found"] > 0).sum()
        
        kpis = {
            "Total Segments": f"{total_segments}",
            "% with Strong Trend": f"{(strong_trend_count / total_segments * 100):.1f}%",
            "% with Strong Seasonality": f"{(strong_season_count / total_segments * 100):.1f}%",
            "% with Anomalies": f"{(anomaly_count / total_segments * 100):.1f}%"
        }
        
        for col, (name, value) in enumerate(kpis.items()):
            card = QFrame()
            card.setObjectName("card")
            card.setFixedHeight(150)  # Fixed height for KPI cards
            card.setMinimumWidth(150)  # Minimum width
            card_layout = QVBoxLayout(card)
            card_layout.setAlignment(Qt.AlignCenter)
            
            value_label = QLabel(value)
            value_label.setObjectName("h1_label")
            value_label.setAlignment(Qt.AlignCenter)
            
            name_label = QLabel(name)
            name_label.setObjectName("stats_label")
            name_label.setAlignment(Qt.AlignCenter)
            name_label.setWordWrap(True)
            
            card_layout.addWidget(value_label)
            card_layout.addWidget(name_label)
            kpi_layout.addWidget(card, 0, col)
        
        layout.addLayout(kpi_layout)

        # Summary Charts and Top Lists with proper sizing
        insights_layout = QHBoxLayout()
        charts_widget = self._create_summary_charts(summary_df)
        charts_widget.setMinimumHeight(300)
        charts_widget.setMinimumWidth(400)
        
        top_performers_widget = self._create_top_performers_card(summary_df)
        top_performers_widget.setMinimumHeight(300)
        top_performers_widget.setMinimumWidth(300)
        
        insights_layout.addWidget(charts_widget, 2)  # Give more space to charts
        insights_layout.addWidget(top_performers_widget, 1)
        layout.addLayout(insights_layout)

        # Table with proper sizing
        table_card = QFrame()
        table_card.setObjectName("card")
        table_card.setMinimumHeight(300)
        table_layout = QVBoxLayout(table_card)
        table_layout.addWidget(QLabel("<h3>Segment Details</h3>"))
        
        display_df = summary_df.copy()
        display_df["Trend Strength"] = display_df["Trend Strength"].map('{:.1%}'.format)
        display_df["Seasonality Strength"] = display_df["Seasonality Strength"].map('{:.1%}'.format)
        
        table = self.create_table_widget(display_df, stretch_columns=["Segment"])
        
        # Fix table column widths
        table.setMinimumHeight(200)
        table.horizontalHeader().setStretchLastSection(True)
        for i in range(table.columnCount()):
            if i == 0:  # Segment column
                table.setColumnWidth(i, 200)
            else:
                table.setColumnWidth(i, 120)
        
        table_layout.addWidget(table)
        layout.addWidget(table_card)
        
        return widget

    def _create_summary_charts(self, summary_df: pd.DataFrame):
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Bar chart for Top 10 Trend Strength with fixed size
        top_trends = summary_df.nlargest(10, 'Trend Strength')
        canvas_bar = MatplotlibCanvas(self, height=6, width=8)  # Fixed dimensions
        ax_bar = canvas_bar.fig.add_subplot(111)
        
        if not top_trends.empty:
            sns.barplot(data=top_trends, y='Segment', x='Trend Strength', ax=ax_bar, 
                       palette='viridis', hue='Segment', legend=False)
            ax_bar.set_title("Top Segments by Trend Strength", fontsize=12, pad=10)
            ax_bar.set_xlabel("Strength")
            ax_bar.set_ylabel("")
            ax_bar.set_xlim(0, 1)
            
            # Improve label visibility
            ax_bar.tick_params(axis='y', labelsize=10)
            ax_bar.tick_params(axis='x', labelsize=10)
        else:
            ax_bar.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                       transform=ax_bar.transAxes, fontsize=12)
        
        canvas_bar.fig.tight_layout(pad=2.0)
        layout.addWidget(canvas_bar)
        
        return card

    def _create_top_performers_card(self, summary_df: pd.DataFrame):
        card = QFrame()
        card.setObjectName("card")
        card.setMinimumWidth(250)
        layout = QVBoxLayout(card)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)
        
        top_trend = summary_df.nlargest(3, 'Trend Strength')['Segment'].tolist()
        top_season = summary_df.nlargest(3, 'Seasonality Strength')['Segment'].tolist()
        top_anomalies = summary_df.nlargest(3, 'Anomalies Found')['Segment'].tolist()
        
        title_label = QLabel("<h4>Key Segments</h4>")
        title_label.setAlignment(Qt.AlignTop)
        layout.addWidget(title_label)
        
        def create_list(title_text, items):
            v_layout = QVBoxLayout()
            title = QLabel(f"<b>{title_text}</b>")
            title.setStyleSheet("margin-bottom: 5px;")
            v_layout.addWidget(title)
            
            if not items:
                item_label = QLabel("  - N/A")
                item_label.setStyleSheet("color: #666; margin-left: 10px;")
                v_layout.addWidget(item_label)
            else:
                for item in items:
                    item_label = QLabel(f"  - {item}")
                    item_label.setStyleSheet("margin-left: 10px; margin-bottom: 2px;")
                    item_label.setWordWrap(True)
                    v_layout.addWidget(item_label)
            return v_layout

        layout.addLayout(create_list("Strongest Trend", top_trend))
        layout.addLayout(create_list("Highest Seasonality", top_season))
        layout.addLayout(create_list("Most Anomalies", top_anomalies))
        layout.addStretch()

        return card

    def perform_segment_analysis(self, segment_df, name, target_col):
        if len(segment_df) < 15: 
            card = QFrame()
            card.setObjectName("card_light")
            card.setMinimumHeight(100)
            layout = QVBoxLayout(card)
            layout.addWidget(QLabel(f"<b>Segment: {name}</b>"))
            layout.addWidget(QLabel("Not enough data points (minimum 15) for detailed analysis."))
            return card

        card = QFrame()
        card.setObjectName("card")
        card.setMinimumHeight(700)  # Ensure minimum height
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(15, 15, 15, 15)
        
        title = QLabel(f"<h3>Detailed Analysis for Segment: {name}</h3>")
        card_layout.addWidget(title)
        
        sub_tabs = QTabWidget()
        sub_tabs.setMinimumHeight(600)  # Minimum height for sub tabs
        card_layout.addWidget(sub_tabs)
        
        sub_tabs.addTab(self._create_temporal_patterns_widget(segment_df, target_col), "Temporal Patterns")
        sub_tabs.addTab(self._create_anomalies_widget(segment_df, target_col), "Anomalies")
        sub_tabs.addTab(self._create_correlation_widget(segment_df), "Correlations")
        
        return card

    def _create_temporal_patterns_widget(self, df, target_col):
        widget = QWidget()
        widget.setMinimumHeight(500)
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        
        time_series = df.set_index(self.date_col)[target_col].sort_index()
        
        try:
            period_val = min(12, len(time_series) // 2)
            if period_val < 2:
                error_label = QLabel("Not enough data points for seasonal decomposition.")
                error_label.setStyleSheet("font-size: 12px; color: #666; padding: 20px;")
                layout.addWidget(error_label)
                return widget

            result = seasonal_decompose(time_series, model='additive', period=period_val, extrapolate_trend='freq')
            
            # Create decomposition plot with fixed size
            canvas = MatplotlibCanvas(self, height=10, width=12)  # Fixed larger size
            axs = canvas.fig.subplots(4, 1, sharex=True)
            
            result.observed.plot(ax=axs[0], title="Observed", color="#4C6EF5", lw=1.5)
            axs[0].grid(True, alpha=0.3)
            axs[0].set_ylabel("Value")
            
            result.trend.plot(ax=axs[1], title="Trend", color="#4C6EF5", lw=1.5)
            axs[1].grid(True, alpha=0.3)
            axs[1].set_ylabel("Trend")
            
            result.seasonal.plot(ax=axs[2], title="Seasonal", color="#4C6EF5", lw=1.5)
            axs[2].grid(True, alpha=0.3)
            axs[2].set_ylabel("Seasonal")
            
            result.resid.plot(ax=axs[3], marker='o', linestyle='None', markersize=4, 
                             color="#4C6EF5", alpha=0.6, title="Residuals")
            axs[3].grid(True, alpha=0.3)
            axs[3].set_ylabel("Residuals")
            axs[3].set_xlabel("Date")
            
            canvas.fig.suptitle("Time Series Decomposition", fontsize=14, y=0.98)
            canvas.fig.tight_layout(pad=3.0, rect=[0, 0, 1, 0.96])
            
            layout.addWidget(QLabel("<h4>Time Series Decomposition</h4>"))
            layout.addWidget(canvas)
            
        except Exception as e: 
            error_label = QLabel(f"<b>Decomposition Error:</b> {str(e)}")
            error_label.setStyleSheet("color: #d32f2f; padding: 10px;")
            layout.addWidget(error_label)
        
        try:
            # ACF/PACF plots with fixed size
            canvas2 = MatplotlibCanvas(self, height=6, width=12)
            ax1 = canvas2.fig.add_subplot(121)
            ax2 = canvas2.fig.add_subplot(122)
            
            plot_acf(time_series, ax=ax1, title="Autocorrelation (ACF)", lags=min(40, len(time_series)//4))
            plot_pacf(time_series, ax=ax2, title="Partial Autocorrelation (PACF)", lags=min(40, len(time_series)//4))
            
            ax1.grid(True, alpha=0.3)
            ax2.grid(True, alpha=0.3)
            
            canvas2.fig.tight_layout(pad=3.0)
            layout.addWidget(QLabel("<h4>ACF & PACF Plots</h4>"))
            layout.addWidget(canvas2)
            
        except Exception as e: 
            error_label = QLabel(f"<b>ACF/PACF Error:</b> {str(e)}")
            error_label.setStyleSheet("color: #d32f2f; padding: 10px;")
            layout.addWidget(error_label)
            
        return widget

    def _create_anomalies_widget(self, df, target_col):
        widget = QWidget()
        widget.setMinimumHeight(400)
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        
        try:
            series = df.set_index(self.date_col)[target_col].dropna()
            if len(series) < 2:
                error_label = QLabel("Not enough data points for anomaly detection.")
                error_label.setStyleSheet("font-size: 12px; color: #666; padding: 20px;")
                layout.addWidget(error_label)
                return widget

            z_scores = np.abs(zscore(series))
            anomalies = series[z_scores > 3]
            
            # Create anomaly plot with fixed size
            canvas = MatplotlibCanvas(self, height=6, width=12)
            ax = canvas.fig.add_subplot(111)
            
            ax.plot(series.index, series.values, label='Time Series', zorder=1, 
                   color="#4C6EF5", linewidth=1.5)
            
            if not anomalies.empty:
                ax.scatter(anomalies.index, anomalies.values, color='red', 
                          label=f'Anomalies ({len(anomalies)})', zorder=2, s=50, alpha=0.8)
            
            ax.set_title(f"Anomaly Detection in '{target_col}'", fontsize=14, pad=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Date")
            ax.set_ylabel("Value")
            
            canvas.fig.tight_layout(pad=2.0)
            layout.addWidget(canvas)
            
            # Add anomaly summary
            if not anomalies.empty:
                summary_label = QLabel(f"<b>Summary:</b> Found {len(anomalies)} anomalies "
                                     f"({len(anomalies)/len(series)*100:.1f}% of data points)")
                summary_label.setStyleSheet("padding: 10px; background: #f5f5f5; border-radius: 5px;")
                layout.addWidget(summary_label)
            
        except Exception as e:
            error_label = QLabel(f"<b>Error creating anomaly plot:</b> {str(e)}")
            error_label.setStyleSheet("color: #d32f2f; padding: 10px;")
            layout.addWidget(error_label)
            
        return widget

    def _create_correlation_widget(self, df):
        widget = QWidget()
        widget.setMinimumHeight(400)
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        
        numeric_df = df.select_dtypes(include=np.number)
        if len(numeric_df.columns) < 2:
            error_label = QLabel("Not enough numeric columns for correlation analysis.")
            error_label.setStyleSheet("font-size: 12px; color: #666; padding: 20px;")
            layout.addWidget(error_label)
            return widget
            
        try:
            # Create correlation heatmap with fixed size
            canvas = MatplotlibCanvas(self, height=8, width=10)
            ax = canvas.fig.add_subplot(111)
            
            corr_matrix = numeric_df.corr()
            
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax, 
                       annot_kws={"size": 10}, cbar_kws={"shrink": 0.8})
            
            ax.set_title("Correlation Matrix", fontsize=14, pad=10)
            
            # Rotate labels for better readability
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            
            canvas.fig.tight_layout(pad=2.0)
            layout.addWidget(canvas)
            
        except Exception as e:
            error_label = QLabel(f"<b>Error creating correlation matrix:</b> {str(e)}")
            error_label.setStyleSheet("color: #d32f2f; padding: 10px;")
            layout.addWidget(error_label)
            
        return widget
    
# --- Analysis Page 2: Demand Patterns (Integrate ABC-COV) ---
class DemandPatternsPage(BaseEdaPage):
    def __init__(self, app_state):
        super().__init__(app_state)
        self.title_label.setText("Demand Pattern Analysis") # More general title

        # Use a QTabWidget for ADI/CV2 and ABC-COV
        self.demand_tabs = QTabWidget()
        self.plot_layout.addWidget(self.demand_tabs)

        # ADI/CV2 Sub-Tab
        self.adi_cov2_widget = QWidget()
        self.adi_cov2_layout = QVBoxLayout(self.adi_cov2_widget)
        self.adi_cov2_layout.setContentsMargins(0,0,0,0) # No extra margins
        self.adi_cov2_layout.setAlignment(Qt.AlignTop)

        # --- ADI/CV2 Controls and Period Selection ---
        adi_controls_group = QFrame(); adi_controls_group.setObjectName("card");
        adi_controls_layout = QHBoxLayout(adi_controls_group)
        adi_controls_layout.setSpacing(10)

        adi_controls_layout.addWidget(QLabel("Consider Last:"))
        self.adi_num_periods_spinbox = QSpinBox()
        self.adi_num_periods_spinbox.setRange(1, 1000)
        self.adi_num_periods_spinbox.setValue(12) # Default 12 months (1 year)
        adi_controls_layout.addWidget(self.adi_num_periods_spinbox)

        self.adi_period_unit_combo = QComboBox()
        self.adi_period_unit_combo.addItems(["Months", "Years"])
        self.adi_period_unit_combo.setCurrentText("Months")
        adi_controls_layout.addWidget(self.adi_period_unit_combo)

        self.adi_run_button = QPushButton("Generate ADI/CV² Analysis")
        self.adi_run_button.setMinimumWidth(200)
        self.adi_run_button.clicked.connect(self._run_adi_cov2_analysis)
        adi_controls_layout.addWidget(self.adi_run_button)
        adi_controls_layout.addStretch()
        self.adi_cov2_layout.addWidget(adi_controls_group)


        self.adi_cov2_plot_area = QScrollArea()
        self.adi_cov2_plot_area.setWidgetResizable(True)
        self.adi_cov2_plot_area.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        self.adi_cov2_plot_container = QWidget()
        self.adi_cov2_plot_container.setStyleSheet("background-color: transparent;")
        self.adi_cov2_inner_plot_layout = QVBoxLayout(self.adi_cov2_plot_container)
        self.adi_cov2_inner_plot_layout.setAlignment(Qt.AlignTop)
        self.adi_cov2_inner_plot_layout.setSpacing(25)
        self.adi_cov2_plot_area.setWidget(self.adi_cov2_plot_container)
        self.adi_cov2_layout.addWidget(self.adi_cov2_plot_area, 1)

        self.demand_tabs.addTab(self.adi_cov2_widget, "ADI/CV² Segmentation")


        # ABC-COV Sub-Tab
        self.abc_cov_widget = QWidget()
        self.abc_cov_layout = QVBoxLayout(self.abc_cov_widget)
        self.abc_cov_layout.setContentsMargins(0,0,0,0)
        self.abc_cov_layout.setAlignment(Qt.AlignTop)

        abc_cov_controls_group = QFrame(); abc_cov_controls_group.setObjectName("card")
        self.abc_cov_controls_grid = QGridLayout(abc_cov_controls_group)
        self.abc_cov_controls_grid.setSpacing(10)

        # Row 0: Dimension, Measure, ABC Toggle, COV Toggle
        self.abc_cov_controls_grid.addWidget(QLabel("Dimension Column:"), 0, 0)
        self.abc_cov_dimension_combo = QComboBox()
        self.abc_cov_dimension_combo.setMinimumWidth(150)
        self.abc_cov_controls_grid.addWidget(self.abc_cov_dimension_combo, 0, 1)

        self.abc_cov_controls_grid.addWidget(QLabel("Measure Column:"), 1, 0)
        self.abc_cov_measure_combo = QComboBox()
        self.abc_cov_measure_combo.setMinimumWidth(150)
        self.abc_cov_controls_grid.addWidget(self.abc_cov_measure_combo, 1, 1)

        self.apply_abc_checkbox = QCheckBox("Apply ABC Segmentation")
        self.apply_abc_checkbox.setChecked(True)
        self.apply_abc_checkbox.toggled.connect(self._toggle_abc_thresholds)
        self.abc_cov_controls_grid.addWidget(self.apply_abc_checkbox, 0, 2, 1, 4) # Span 4 columns

        self.apply_cov_checkbox = QCheckBox("Apply COV Segmentation")
        self.apply_cov_checkbox.setChecked(True)
        self.apply_cov_checkbox.toggled.connect(self._toggle_cov_thresholds)
        self.abc_cov_controls_grid.addWidget(self.apply_cov_checkbox, 0, 6, 1, 4) # Span 4 columns

        # Row 1: ABC Class A, COV Class X
        self.abc_cov_controls_grid.addWidget(QLabel("Class A (%):"), 1, 2)
        self.abc_a_op_combo = QComboBox(); self.abc_a_op_combo.addItems([">=", "<=", ">", "<"]); self.abc_a_op_combo.setCurrentText("<=")
        self.abc_cov_controls_grid.addWidget(self.abc_a_op_combo, 1, 3)
        self.abc_a_threshold = QLineEdit("80")
        self.abc_a_threshold.setValidator(self._create_percentage_validator())
        self.abc_cov_controls_grid.addWidget(self.abc_a_threshold, 1, 4)

        self.abc_cov_controls_grid.addWidget(QLabel("Class X (Val):"), 1, 6)
        self.cov_x_op_combo = QComboBox(); self.cov_x_op_combo.addItems([">=", "<=", ">", "<"]); self.cov_x_op_combo.setCurrentText("<=")
        self.cov_x_threshold = QLineEdit("0.5")
        self.cov_x_threshold.setValidator(self._create_float_validator())
        self.abc_cov_controls_grid.addWidget(self.cov_x_threshold, 1, 8)

        # Row 2: ABC Class B, COV Class Y
        self.abc_cov_controls_grid.addWidget(QLabel("Class B (%):"), 2, 2)
        self.abc_b_op_combo = QComboBox(); self.abc_b_op_combo.addItems([">=", "<=", ">", "<"]); self.abc_b_op_combo.setCurrentText("<=")
        self.abc_b_threshold = QLineEdit("95")
        self.abc_b_threshold.setValidator(self._create_percentage_validator())
        self.abc_cov_controls_grid.addWidget(self.abc_b_threshold, 2, 4)

        self.abc_cov_controls_grid.addWidget(QLabel("Class Y (Val):"), 2, 6)
        self.cov_y_op_combo = QComboBox(); self.cov_y_op_combo.addItems([">=", "<=", ">", "<"]); self.cov_y_op_combo.setCurrentText("<=")
        self.cov_y_threshold = QLineEdit("1.0")
        self.cov_y_threshold.setValidator(self._create_float_validator())
        self.abc_cov_controls_grid.addWidget(self.cov_y_threshold, 2, 8)

        # Row 3: ABC Class C and COV Class Z (Full definition now)
        self.abc_cov_controls_grid.addWidget(QLabel("Class C (%):"), 3, 2)
        self.abc_c_op_combo = QComboBox(); self.abc_c_op_combo.addItems([">=", "<=", ">", "<", "SKIP"]); self.abc_c_op_combo.setCurrentText(">") # Typically > B threshold
        self.abc_cov_controls_grid.addWidget(self.abc_c_op_combo, 3, 3)
        self.abc_c_threshold = QLineEdit("95") # Default to B threshold as C starts after B
        self.abc_c_threshold.setValidator(self._create_percentage_validator())
        self.abc_cov_controls_grid.addWidget(self.abc_c_threshold, 3, 4)

        self.abc_cov_controls_grid.addWidget(QLabel("Class Z (Val):"), 3, 6)
        self.cov_z_op_combo = QComboBox(); self.cov_z_op_combo.addItems([">=", "<=", ">", "<", "SKIP"]); self.cov_z_op_combo.setCurrentText(">") # Typically > Y threshold
        self.cov_z_threshold = QLineEdit("1.0") # Default to Y threshold as Z starts after Y
        self.cov_z_threshold.setValidator(self._create_float_validator())
        self.abc_cov_controls_grid.addWidget(self.cov_z_threshold, 3, 8)


        # Row 4: Generate Button
        self.abc_cov_run_button = QPushButton("Generate Segmentation")
        self.abc_cov_run_button.clicked.connect(self._run_abc_cov_analysis)
        self.abc_cov_controls_grid.addWidget(self.abc_cov_run_button, 4, 0, 1, -1) # Spans all columns


        self.abc_cov_layout.addWidget(abc_cov_controls_group)
        self.abc_cov_layout.addStretch()

        self.abc_cov_plot_area = QScrollArea()
        self.abc_cov_plot_area.setWidgetResizable(True)
        self.abc_cov_plot_area.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        self.abc_cov_plot_container = QWidget()
        self.abc_cov_plot_container.setStyleSheet("background-color: transparent;")
        self.abc_cov_inner_plot_layout = QVBoxLayout(self.abc_cov_plot_container)
        self.abc_cov_inner_plot_layout.setAlignment(Qt.AlignTop)
        self.abc_cov_inner_plot_layout.setSpacing(25)
        self.abc_cov_plot_area.setWidget(self.abc_cov_plot_container)
        self.abc_cov_layout.addWidget(self.abc_cov_plot_area, 1)

        self.demand_tabs.addTab(self.abc_cov_widget, "ABC-COV Segmentation")

        # Initial toggle states
        self._toggle_abc_thresholds(self.apply_abc_checkbox.isChecked())
        self._toggle_cov_thresholds(self.apply_cov_checkbox.isChecked())


    def _create_percentage_validator(self):
        validator = QDoubleValidator(0.0, 100.0, 2)
        validator.setNotation(QDoubleValidator.StandardNotation)
        return validator

    def _create_float_validator(self):
        validator = QDoubleValidator()
        validator.setNotation(QDoubleValidator.StandardNotation)
        return validator

    def _toggle_abc_thresholds(self, checked):
        # Enables/disables ABC related controls
        self.abc_a_op_combo.setEnabled(checked)
        self.abc_a_threshold.setEnabled(checked)
        self.abc_b_op_combo.setEnabled(checked)
        self.abc_b_threshold.setEnabled(checked)
        self.abc_c_op_combo.setEnabled(checked)
        self.abc_c_threshold.setEnabled(checked)

    def _toggle_cov_thresholds(self, checked):
        # Enables/disables COV related controls
        self.cov_x_op_combo.setEnabled(checked)
        self.cov_x_threshold.setEnabled(checked)
        self.cov_y_op_combo.setEnabled(checked)
        self.cov_y_threshold.setEnabled(checked)
        self.cov_z_op_combo.setEnabled(checked)
        self.cov_z_threshold.setEnabled(checked)


    def update_controls(self):
        """Updates controls for both ADI/CV2 and ABC-COV sub-tabs."""
        # For ABC-COV
        self.abc_cov_dimension_combo.clear()
        self.abc_cov_measure_combo.clear()

        if self.df is None:
            return

        all_dims = self.grouping_cols + [c for c in self.categorical_cols if c not in self.grouping_cols]
        if all_dims:
            self.abc_cov_dimension_combo.addItems(sorted(all_dims))

        if self.numeric_cols:
            self.abc_cov_measure_combo.addItems(sorted(self.numeric_cols))
            priority_targets = [col for col in self.numeric_cols if any(k in col.lower() for k in self.PRIORITY_KEYWORDS)]
            if priority_targets:
                self.abc_cov_measure_combo.setCurrentText(priority_targets[0])
            elif self.numeric_cols:
                self.abc_cov_measure_combo.setCurrentText(self.numeric_cols[0])


    def _clear_adi_cov2_plots(self):
        while self.adi_cov2_inner_plot_layout.count():
            child = self.adi_cov2_inner_plot_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()

    def _clear_abc_cov_plots(self):
        while self.abc_cov_inner_plot_layout.count():
            child = self.abc_cov_inner_plot_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()

    # --- ADI/CV2 Analysis Logic (Modified for dynamic periods) ---
    def _run_adi_cov2_analysis(self):
        self._clear_adi_cov2_plots() # Clear only ADI/CV2 plots

        df = self.app_state.aggregated_df

        if df is None or self.date_col is None or not self.grouping_cols:
            self.adi_cov2_inner_plot_layout.addWidget(QLabel("<h3>Analysis Requirement</h3><p>This analysis requires data to be aggregated by at least one dimension in the 'Aggregation' tab and a date column.</p>"))
            return

        target_col = self.numeric_cols[0] if self.numeric_cols else None
        if not target_col:
            self.adi_cov2_inner_plot_layout.addWidget(QLabel("<h3>Error</h3><p>No numeric columns found in the aggregated data to analyze.</p>"))
            return

        num_periods = self.adi_num_periods_spinbox.value()
        period_unit = self.adi_period_unit_combo.currentText()

        # Filter data for the last N periods/years
        df[self.date_col] = pd.to_datetime(df[self.date_col]) # Ensure date_col is datetime type
        max_date = df[self.date_col].max()

        start_date_filter = None
        if period_unit == "Months":
            # Corrected: Use DateOffset without -1 for months to include the last 'num_periods'
            start_date_filter = max_date - pd.DateOffset(months=num_periods)
        elif period_unit == "Years":
            # Corrected: Use DateOffset without -1 for years to include the last 'num_periods'
            start_date_filter = max_date - pd.DateOffset(years=num_periods)

        if start_date_filter:
            # Adjust start_date_filter to be the beginning of the month/year
            if period_unit == "Months":
                start_date_filter = start_date_filter.normalize().replace(day=1)
            elif period_unit == "Years":
                start_date_filter = start_date_filter.normalize().replace(month=1, day=1)

            df_filtered = df[df[self.date_col] >= start_date_filter].copy()
            if df_filtered.empty:
                self.adi_cov2_inner_plot_layout.addWidget(QLabel(f"<h3>Error: No data found for the last {num_periods} {period_unit}.</h3><p>Try a different period selection.</p>"))
                return
        else:
            df_filtered = df.copy() # If no filtering applied (e.g., initial state)


        try:
            # Re-calculate full_date_range based on filtered data
            full_date_range = pd.date_range(start=df_filtered[self.date_col].min(), end=df_filtered[self.date_col].max(), freq='MS')
            results = []

            for name, group in df_filtered.groupby(self.grouping_cols):
                ts = group.set_index(self.date_col)[target_col]
                ts_full = ts.reindex(full_date_range, fill_value=0)
                non_zero = ts_full[ts_full > 0]

                if len(non_zero) <= 1 or np.mean(non_zero) == 0: continue

                adi = len(ts_full) / len(non_zero)
                cv2 = (np.std(non_zero) / np.mean(non_zero)) ** 2 if np.mean(non_zero) != 0 else 0

                results.append({
                    'group': " | ".join(map(str, name)) if isinstance(name, tuple) else name,
                    'ADI': adi, 'CV2': cv2, 'Total_Value': ts_full.sum()
                })

            if not results:
                self.adi_cov2_inner_plot_layout.addWidget(QLabel("Not enough data to calculate ADI/CV² for any segment within the selected period."))
                return

            results_df = pd.DataFrame(results)
            results_df['Demand_Type'] = np.select(
                [(results_df['ADI'] >= 1.32) & (results_df['CV2'] >= 0.49), (results_df['ADI'] < 1.32) & (results_df['CV2'] >= 0.49), (results_df['ADI'] >= 1.32) & (results_df['CV2'] < 0.49)],
                ['Lumpy', 'Erratic', 'Intermittent'], default='Smooth'
            )

            card = QFrame(); card.setObjectName("card"); card_layout = QVBoxLayout(card); card_layout.setSpacing(20)
            card_layout.addWidget(QLabel(f"<h3>Demand Classification Matrix (ADI/CV²) for last {num_periods} {period_unit}</h3>"))

            plot_canvas = MatplotlibCanvas(self, height=7)
            plot_canvas.setMinimumHeight(500)
            ax = plot_canvas.fig.add_subplot(111)
            sns.scatterplot(data=results_df, x='ADI', y='CV2', hue='Demand_Type', ax=ax, s=60, alpha=0.8, palette="Set2")
            ax.axvline(x=1.32, c='black', ls='--', lw=1.2, alpha=0.8); ax.axhline(y=0.49, c='black', ls='--', lw=1.2, alpha=0.8)
            ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel("ADI (log scale)"); ax.set_ylabel("CV² (log scale)")
            ax.set_title("Demand Pattern Segmentation"); plot_canvas.fig.tight_layout(); card_layout.addWidget(plot_canvas)
            card_layout.addWidget(self.create_separator())

            card_layout.addWidget(QLabel("<h3>Summary & Segment Details</h3>"))
            summary_tabs = QTabWidget()
            summary_tabs.addTab(self._create_summary_page_adi_cov2(results_df, target_col), "📊 Overall Summary")
            for demand_type in sorted(results_df['Demand_Type'].unique()):
                type_df = results_df[results_df['Demand_Type'] == demand_type]
                summary_tabs.addTab(self._create_bifurcation_page_adi_cov2(type_df), f"{demand_type} Segments ({len(type_df)})")
            card_layout.addWidget(summary_tabs)
            self.adi_cov2_inner_plot_layout.addWidget(card)

        except Exception as e:
            self.adi_cov2_inner_plot_layout.addWidget(QLabel(f"<b>Demand Pattern Analysis Error:</b> {e}"))
            print(f"ADI/CV2 Analysis Error: {e}")

    def _create_summary_page_adi_cov2(self, results_df, target_col):
        widget = QWidget(); layout = QHBoxLayout(widget); layout.setSpacing(25)
        summary_stats = results_df.groupby('Demand_Type')['Total_Value'].agg(['sum', 'count'])
        total_sum = summary_stats['sum'].sum(); total_count = summary_stats['count'].sum()
        summary_stats['Value %'] = (summary_stats['sum'] / total_sum * 100).round(2)
        summary_stats['Segment %'] = (summary_stats['count'] / total_count * 100).round(2)
        summary_stats.rename(columns={'sum': f'Total {target_col}', 'count': '# of Segments'}, inplace=True)

        table = QTableWidget()
        table.setRowCount(len(summary_stats))
        table.setColumnCount(len(summary_stats.columns) + 1)
        table.setHorizontalHeaderLabels(['Demand Type'] + list(summary_stats.columns))
        table.setSortingEnabled(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        for i, (idx, row) in enumerate(summary_stats.iterrows()):
            table.setItem(i, 0, QTableWidgetItem(str(idx)))
            for j, col_name in enumerate(summary_stats.columns):
                item = QTableWidgetItem(f"{row[col_name]:,.2f}")
                item.setTextAlignment(Qt.AlignRight|Qt.AlignVCenter)
                table.setItem(i, j + 1, item)

        table.resizeColumnsToContents()
        table.setMinimumHeight(300) # Will resize down if content is smaller than DEFAULT_TABLE_ROW_HEIGHT
        layout.addWidget(table, 3)

        canvas = MatplotlibCanvas(self, height=4); ax = canvas.fig.add_subplot(111)
        summary_stats.plot(kind='pie', y='Value %', labels=summary_stats.index, autopct='%1.1f%%', ax=ax, legend=False)
        ax.set_ylabel(''); ax.set_title(f'% Contribution of Total {target_col}'); canvas.fig.tight_layout(); layout.addWidget(canvas, 2)
        return widget

    def _create_bifurcation_page_adi_cov2(self, type_df):
        widget = QWidget(); layout = QVBoxLayout(widget); table = QTableWidget()
        table.setColumnCount(3); table.setHorizontalHeaderLabels(['Segment Name', 'ADI', 'CV²']); table.setRowCount(len(type_df)); table.setSortingEnabled(True)
        # Set fixed row height
        table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        table.verticalHeader().setDefaultSectionSize(DEFAULT_TABLE_ROW_HEIGHT)

        for i, row in type_df.iterrows():
            table.setItem(i, 0, QTableWidgetItem(row['group'])); table.setItem(i, 1, QTableWidgetItem(f"{row['ADI']:.2f}")); table.setItem(i, 2, QTableWidgetItem(f"{row['CV2']:.2f}"))

        header = table.horizontalHeader()
        for i in range(header.count()):
            header.setSectionResizeMode(i, QHeaderView.Stretch)
        table.resizeRowsToContents() # Will resize down if content is smaller than DEFAULT_TABLE_ROW_HEIGHT
        table.setMinimumHeight(300)
        layout.addWidget(table)
        return widget

    # --- ABC-COV Analysis Logic (INTEGRATED AND ENHANCED) ---
    def _run_abc_cov_analysis(self):
        self._clear_abc_cov_plots() # Clear only ABC-COV plots

        df = self.app_state.aggregated_df
        dimension_col = self.abc_cov_dimension_combo.currentText()
        measure_col = self.abc_cov_measure_combo.currentText()

        apply_abc = self.apply_abc_checkbox.isChecked()
        apply_cov = self.apply_cov_checkbox.isChecked()

        if df is None or not dimension_col or not measure_col:
            self.abc_cov_inner_plot_layout.addWidget(QLabel("<h3>Please select a Dimension and a Measure to analyze.</h3><p>Ensure data is aggregated first.</p>"))
            return
        if not apply_abc and not apply_cov:
            self.abc_cov_inner_plot_layout.addWidget(QLabel("<h3>Please select at least one segmentation method (ABC or COV).</h3>"))
            return


        try:
            # Get thresholds and operators
            abc_a_perc = float(self.abc_a_threshold.text()) / 100
            abc_b_perc = float(self.abc_b_threshold.text()) / 100
            abc_c_perc = float(self.abc_c_threshold.text()) / 100 # NEW C THRESHOLD
            
            cov_x_val = float(self.cov_x_threshold.text())
            cov_y_val = float(self.cov_y_threshold.text())
            cov_z_val = float(self.cov_z_threshold.text()) # NEW Z THRESHOLD

            abc_a_op = self.abc_a_op_combo.currentText()
            abc_b_op = self.abc_b_op_combo.currentText()
            abc_c_op = self.abc_c_op_combo.currentText() # NEW C OPERATOR
            
            cov_x_op = self.cov_x_op_combo.currentText()
            cov_y_op = self.cov_y_op_combo.currentText()
            cov_z_op = self.cov_z_op_combo.currentText() # NEW Z OPERATOR

            # --- Validation ---
            if apply_abc:
                if not (0 <= abc_a_perc <= 1 and 0 <= abc_b_perc <= 1 and 0 <= abc_c_perc <= 1):
                     self.abc_cov_inner_plot_layout.addWidget(QLabel("<h3>Error: Invalid ABC percentage thresholds.</h3><p>Ensure 0 <= A, B, C <= 100.</p>"))
                     return
                # Basic logical order validation (more complex validation for all operator combinations is out of scope here)
                # It's better to warn the user about potentially illogical setups rather than halt execution.
                if (abc_a_perc >= abc_b_perc and abc_b_op == "<=") or \
                   (abc_b_perc >= abc_c_perc and abc_c_op == "<="):
                    self.abc_cov_inner_plot_layout.addWidget(QLabel("<h3>Warning: ABC thresholds may be illogical. A < B < C is recommended for cumulative percentages with '<=' operator.</h3>"))


            if apply_cov:
                if not (0 <= cov_x_val and 0 <= cov_y_val and 0 <= cov_z_val):
                     self.abc_cov_inner_plot_layout.addWidget(QLabel("<h3>Error: Invalid COV thresholds.</h3><p>Ensure 0 <= X, Y, Z.</p>"))
                     return
                # Basic logical order validation
                if (cov_x_val >= cov_y_val and cov_y_op == "<=") or \
                   (cov_y_val >= cov_z_val and cov_z_op == "<="):
                     self.abc_cov_inner_plot_layout.addWidget(QLabel("<h3>Warning: COV thresholds may be illogical. X < Y < Z is recommended for variability with '<=' operator.</h3>"))


            # --- Calculate Total Measure and COV for each dimension item ---
            analysis_df = df.groupby(dimension_col).agg(
                total_measure=(measure_col, 'sum'),
                std_measure=(measure_col, 'std'),
                mean_measure=(measure_col, 'mean')
            ).reset_index()

            analysis_df['COV'] = analysis_df.apply(
                lambda row: row['std_measure'] / row['mean_measure'] if row['mean_measure'] != 0 and pd.notna(row['mean_measure']) else 0,
                axis=1
            )
            analysis_df['COV'].fillna(0, inplace=True) # Fill NaN COV (e.g., for single data points or mean=0) with 0

            # --- Generic comparison function ---
            def apply_comparison_op(value, threshold, operator):
                if operator == ">=": return value >= threshold
                if operator == ">": return value > threshold
                if operator == "<=": return value <= threshold
                if operator == "<": return value < threshold
                return False # Should not happen

            # --- ABC Analysis (Volume Segmentation with 3 custom classes) ---
            analysis_df['ABC_Class'] = '' # Initialize with empty string for no classification
            if apply_abc:
                analysis_df = analysis_df.sort_values(by='total_measure', ascending=False).reset_index(drop=True)
                analysis_df['cumulative_percentage'] = (analysis_df['total_measure'].cumsum() / analysis_df['total_measure'].sum())

                abc_classes = []
                for idx, row in analysis_df.iterrows():
                    val = row['cumulative_percentage']
                    
                    if apply_comparison_op(val, abc_a_perc, abc_a_op):
                        abc_classes.append('A')
                    elif apply_comparison_op(val, abc_b_perc, abc_b_op):
                        abc_classes.append('B')
                    # Only apply C if operator is NOT "SKIP" and condition is met
                    elif abc_c_op != "SKIP" and apply_comparison_op(val, abc_c_perc, abc_c_op):
                        abc_classes.append('C')
                    else:
                        abc_classes.append('') # Not classified
                analysis_df['ABC_Class'] = abc_classes

            # --- COV Analysis (Variability Segmentation with 3 custom classes) ---
            analysis_df['COV_Class'] = '' # Initialize with empty string for no classification
            if apply_cov:
                cov_classes = []
                for idx, row in analysis_df.iterrows():
                    val = row['COV']
                    if apply_comparison_op(val, cov_x_val, cov_x_op):
                        cov_classes.append('X')
                    elif apply_comparison_op(val, cov_y_val, cov_y_op):
                        cov_classes.append('Y')
                    # Only apply Z if operator is NOT "SKIP" and condition is met
                    elif cov_z_op != "SKIP" and apply_comparison_op(val, cov_z_val, cov_z_op):
                        cov_classes.append('Z')
                    else:
                        cov_classes.append('') # Not classified
                analysis_df['COV_Class'] = cov_classes


            # --- Combined Segmentation ---
            if apply_abc and apply_cov:
                # Combine only if both classes are assigned and not empty
                analysis_df['Segment'] = analysis_df.apply(lambda row: row['ABC_Class'] + row['COV_Class'] if row['ABC_Class'] != '' and row['COV_Class'] != '' else '', axis=1)
                analysis_df_plot = analysis_df[analysis_df['Segment'] != ''].copy() # Filter for plotting
            elif apply_abc:
                analysis_df['Segment'] = analysis_df['ABC_Class']
                analysis_df_plot = analysis_df[analysis_df['Segment'] != ''].copy() # Filter for plotting
            elif apply_cov:
                analysis_df['Segment'] = analysis_df['COV_Class']
                analysis_df_plot = analysis_df[analysis_df['Segment'] != ''].copy() # Filter for plotting
            else:
                analysis_df['Segment'] = '' # No segmentation if neither is applied
                analysis_df_plot = analysis_df[analysis_df['Segment'] != ''].copy() # Filter for plotting (will be empty)

            if analysis_df_plot.empty:
                self.abc_cov_inner_plot_layout.addWidget(QLabel("<h3>No items classified based on the selected criteria and thresholds.</h3><p>Adjust segmentation settings.</p>"))
                return


            # --- Visualization ---
            chart_card = QFrame(); chart_card.setObjectName("card")
            chart_layout = QVBoxLayout(chart_card)
            chart_layout.addWidget(QLabel(f"<h3>Demand Segmentation: {measure_col} vs. COV</h3>"))

            canvas = MatplotlibCanvas(self, height=8)
            ax = canvas.fig.add_subplot(111)

            # Define a consistent color palette for segments (expanded for more potential segments)
            # Using distinct colors for each combined segment
            segment_colors = {
                # Combined (3x3 grid)
                'AX': '#1f77b4', 'AY': '#aec7e8', 'AZ': '#add8e6', # Shades of blue
                'BX': '#ff7f0e', 'BY': '#ffbb78', 'BZ': '#f0e68c', # Shades of orange/yellow
                'CX': '#2ca02c', 'CY': '#98df8a', 'CZ': '#c0d6c0', # Shades of green
                
                # Single-axis (if only one type of segmentation applied)
                'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c', # Distinct primary colors
                'X': '#d62728', 'Y': '#9467bd', 'Z': '#8c564b', # Other distinct colors

                'Unclassified': '#cccccc', # Grey for unclassified
                '': '#cccccc' # Explicitly handle empty segments as grey (not displayed due to filtering)
            }
            # Fallback for any unexpected segment names (ensures all points get a color)
            default_color_idx = 0
            default_colors_cycle = sns.color_palette("tab10") 


            # Plot points manually for precise control and add text labels
            unique_segments = sorted(analysis_df_plot['Segment'].unique()) # Sort for consistent label order
            for segment_name in unique_segments:
                segment_df_plot = analysis_df_plot[segment_name == analysis_df_plot['Segment']] # Corrected filtering
                
                # Get color, cycling through default_colors if segment_name not in segment_colors
                color = segment_colors.get(segment_name)
                if color is None:
                    color = default_colors_cycle[default_color_idx % len(default_colors_cycle)]
                    default_color_idx += 1


                segment_df_plot_pos_vol = segment_df_plot[segment_df_plot['total_measure'] > 0] # Filter out 0/negative for log scale

                if not segment_df_plot_pos_vol.empty:
                    # Scatter plot for points
                    ax.scatter(
                        x=segment_df_plot_pos_vol['total_measure'],
                        y=segment_df_plot_pos_vol['COV'],
                        s=segment_df_plot_pos_vol['total_measure'].apply(lambda x: max(50, min(1000, x / analysis_df_plot['total_measure'].max() * 1000))) ,
                        color=color,
                        alpha=0.7
                    )

                    # Add text label for the segment in the quadrant
                    median_x_data = segment_df_plot_pos_vol['total_measure'].median()
                    median_y_data = segment_df_plot_pos_vol['COV'].median()

                    # Handle cases where median might be NaN or Inf due to data characteristics
                    if not np.isfinite(median_x_data) or median_x_data <= 0:
                        median_x_data = segment_df_plot_pos_vol['total_measure'].mean()
                        if median_x_data <= 0 or not np.isfinite(median_x_data): 
                            # Fallback to a reasonable default if all are zero or NaN
                            median_x_data = analysis_df_plot['total_measure'][analysis_df_plot['total_measure'] > 0].min() if not analysis_df_plot[analysis_df_plot['total_measure'] > 0].empty else 1.0
                    if not np.isfinite(median_y_data):
                        median_y_data = segment_df_plot_pos_vol['COV'].mean()
                        if not np.isfinite(median_y_data): 
                            # Fallback to a reasonable default if all are zero or NaN
                            median_y_data = analysis_df_plot['COV'].mean() if not analysis_df_plot['COV'].empty else 0.1

                    ax.text(
                        median_x_data,
                        median_y_data,
                        segment_name,
                        color='black',
                        fontsize=10,
                        fontweight='bold',
                        ha='center',
                        va='center',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3')
                    )


            # Add Quadrant Lines and labels (More robust placement)
            # ABC (Vertical Lines - Volume Cutoffs)
            if apply_abc:
                sorted_analysis_df = analysis_df.sort_values(by='total_measure', ascending=False)
                if not sorted_analysis_df.empty and sorted_analysis_df['total_measure'].sum() > 0:
                    sorted_analysis_df['cumulative_percentage'] = (sorted_analysis_df['total_measure'].cumsum() / sorted_analysis_df['total_measure'].sum())
                else:
                    sorted_analysis_df['cumulative_percentage'] = 0

                # Function to find the volume value at which a cumulative percentage threshold is crossed
                def get_volume_at_cumulative_perc(target_perc, df_sorted):
                    if df_sorted.empty: return None
                    # Find the first item where cumulative percentage is >= target
                    cutoff_item = df_sorted[df_sorted['cumulative_percentage'] >= target_perc]
                    if cutoff_item.empty: return None # Threshold not reached by any item
                    return cutoff_item['total_measure'].max() # The largest item that gets us to or past the threshold

                a_cutoff_volume = get_volume_at_cumulative_perc(abc_a_perc, sorted_analysis_df)
                b_cutoff_volume = get_volume_at_cumulative_perc(abc_b_perc, sorted_analysis_df)
                c_cutoff_volume = get_volume_at_cumulative_perc(abc_c_perc, sorted_analysis_df) # NEW C CUTOFF

                y_text_pos_top = ax.get_ylim()[1] * 0.95
                
                # Use a set to track drawn lines to avoid duplicates, especially if thresholds are very close
                drawn_abc_lines = set()

                if a_cutoff_volume is not None and a_cutoff_volume > 0:
                    # Round to avoid floating point issues for "distinctness" check
                    rounded_a_vol = round(a_cutoff_volume, 2) 
                    if rounded_a_vol not in drawn_abc_lines:
                        ax.axvline(x=a_cutoff_volume, color='#6C757D', linestyle='--', linewidth=1.2, alpha=0.8)
                        ax.text(a_cutoff_volume, y_text_pos_top, f'ABC-A ({abc_a_perc*100:.0f}%)',
                                color='#212529', ha='center', va='top', fontsize=9, rotation=90,
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
                        drawn_abc_lines.add(rounded_a_vol)

                if b_cutoff_volume is not None and b_cutoff_volume > 0:
                    rounded_b_vol = round(b_cutoff_volume, 2)
                    if rounded_b_vol not in drawn_abc_lines:
                        ax.axvline(x=b_cutoff_volume, color='#6C757D', linestyle='--', linewidth=1.2, alpha=0.8)
                        ax.text(b_cutoff_volume, y_text_pos_top, f'ABC-B ({abc_b_perc*100:.0f}%)',
                                color='#212529', ha='center', va='top', fontsize=9, rotation=90,
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
                        drawn_abc_lines.add(rounded_b_vol)
                
                # NEW C CUTOFF LINE
                if c_cutoff_volume is not None and c_cutoff_volume > 0:
                    rounded_c_vol = round(c_cutoff_volume, 2)
                    if rounded_c_vol not in drawn_abc_lines:
                        ax.axvline(x=c_cutoff_volume, color='#6C757D', linestyle='--', linewidth=1.2, alpha=0.8)
                        ax.text(c_cutoff_volume, y_text_pos_top, f'ABC-C ({abc_c_perc*100:.0f}%)',
                                color='#212529', ha='center', va='top', fontsize=9, rotation=90,
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
                        drawn_abc_lines.add(rounded_c_vol)


            # COV (Horizontal Lines - Variability Cutoffs)
            if apply_cov:
                x_text_pos_right = ax.get_xlim()[1] * 0.99
                
                # Use a set to track drawn lines to avoid duplicates
                drawn_cov_lines = set()

                ax.axhline(y=cov_x_val, color='#4C6EF5', linestyle=':', linewidth=1.2, alpha=0.8)
                ax.text(x_text_pos_right, cov_x_val, f'COV-X ({cov_x_val:.2f})',
                        color='#212529', ha='right', va='center', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
                drawn_cov_lines.add(round(cov_x_val, 2))


                if round(cov_y_val, 2) not in drawn_cov_lines:
                    ax.axhline(y=cov_y_val, color='#4C6EF5', linestyle=':', linewidth=1.2, alpha=0.8)
                    ax.text(x_text_pos_right, cov_y_val, f'COV-Y ({cov_y_val:.2f})',
                            color='#212529', ha='right', va='center', fontsize=9,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
                    drawn_cov_lines.add(round(cov_y_val, 2))
                
                # NEW Z CUTOFF LINE
                if round(cov_z_val, 2) not in drawn_cov_lines:
                    ax.axhline(y=cov_z_val, color='#4C6EF5', linestyle=':', linewidth=1.2, alpha=0.8)
                    ax.text(x_text_pos_right, cov_z_val, f'COV-Z ({cov_z_val:.2f})',
                            color='#212529', ha='right', va='center', fontsize=9,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
                    drawn_cov_lines.add(round(cov_z_val, 2))


            ax.set_xscale('log')
            # Adjust x-axis limits to ensure text labels and points are visible
            if not analysis_df_plot.empty:
                min_x = analysis_df_plot['total_measure'][analysis_df_plot['total_measure'] > 0].min()
                max_x = analysis_df_plot['total_measure'].max()
                
                all_x_vals_for_lim = list(analysis_df_plot['total_measure'].values)
                if apply_abc:
                    if a_cutoff_volume is not None: all_x_vals_for_lim.append(a_cutoff_volume)
                    if b_cutoff_volume is not None: all_x_vals_for_lim.append(b_cutoff_volume)
                    if c_cutoff_volume is not None: all_x_vals_for_lim.append(c_cutoff_volume)
                
                all_x_vals_for_lim = [v for v in all_x_vals_for_lim if v > 0 and pd.notna(v)]

                if all_x_vals_for_lim:
                    min_x_plot = np.min(all_x_vals_for_lim)
                    max_x_plot = np.max(all_x_vals_for_lim)
                    if min_x_plot == max_x_plot: # Handle case of single point or very narrow range
                        ax.set_xlim(min_x_plot * 0.1, max_x_plot * 10)
                    else:
                        ax.set_xlim(min_x_plot * 0.5, max_x_plot * 2) # Expand limits slightly
                else: # Fallback if no valid data for x-limits
                    ax.set_xlim(0.1, 1000) # Default sensible range for log scale


                min_y = analysis_df_plot['COV'].min()
                max_y = analysis_df_plot['COV'].max()

                all_y_vals_for_lim = list(analysis_df_plot['COV'].values)
                if apply_cov:
                    all_y_vals_for_lim.extend([cov_x_val, cov_y_val, cov_z_val])
                
                all_y_vals_for_lim = [v for v in all_y_vals_for_lim if pd.notna(v)]

                if all_y_vals_for_lim:
                    min_y_plot = np.min(all_y_vals_for_lim)
                    max_y_plot = np.max(all_y_vals_for_lim)
                    if min_y_plot == max_y_plot: # Handle case of single point or very narrow range
                        ax.set_ylim(min_y_plot * 0.1 if min_y_plot > 0 else 0, max_y_plot * 10)
                    else:
                        ax.set_ylim(min_y_plot * 0.5 if min_y_plot > 0 else 0, max_y_plot * 1.5) # Expand limits slightly
                else: # Fallback if no valid data for y-limits
                    ax.set_ylim(0, 2.0) # Default sensible range for COV


            ax.set_xlabel(f"Total {measure_col} (Log Scale)", fontsize=12, fontweight='bold', color='#343A40')
            ax.set_ylabel("Coefficient of Variation (COV)", fontsize=12, fontweight='bold', color='#343A40')
            ax.set_title(f"Custom Demand Segmentation for {dimension_col}", fontsize=14, fontweight='bold', color='#212529')
            ax.tick_params(axis='both', which='major', labelsize=10, colors='#495057')
            ax.grid(True, which="both", ls="--", c='#E9ECEF', alpha=0.8) # Lighter grid
            
            # Set background color for the plot area itself (facecolor)
            ax.set_facecolor('#FDFDFD') # Slightly off-white for plot background
            canvas.fig.set_facecolor('#FFFFFF') # White background for the entire figure area

            canvas.fig.tight_layout()

            canvas.setMinimumHeight(600)
            chart_layout.addWidget(canvas)
            self.abc_cov_inner_plot_layout.addWidget(chart_card)

            self.abc_cov_inner_plot_layout.addWidget(self.create_separator())

            # --- Simplified Summary Table ---
            summary_table_card = QFrame(); summary_table_card.setObjectName("card")
            summary_table_layout = QVBoxLayout(summary_table_card)
            summary_table_layout.addWidget(QLabel(f"<h3>Classified Items Details</h3>"))

            summary_table_widget = QTableWidget()
            summary_table_widget.setRowCount(len(analysis_df_plot))
            summary_table_widget.setColumnCount(len(display_cols))
            summary_table_widget.setHorizontalHeaderLabels(display_cols)
            summary_table_widget.setSortingEnabled(True)
            summary_table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
            summary_table_widget.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

            for r_idx, row_data in analysis_df_plot[display_cols].iterrows():
                for c_idx, value in enumerate(row_data):
                    item = QTableWidgetItem(str(value))
                    if display_cols[c_idx] in ['total_measure', 'COV']:
                        item.setTextAlignment(Qt.AlignRight|Qt.AlignVCenter)
                    summary_table_widget.setItem(r_idx, c_idx, item)

            summary_table_widget.resizeColumnsToContents()
            summary_table_widget.setMinimumHeight(400)

            for i in range(summary_table_widget.columnCount()):
                summary_table_widget.horizontalHeader().setSectionResizeMode(i, QHeaderView.Stretch)
            summary_table_widget.setMinimumHeight(min(400, summary_table_widget.verticalHeader().length() + summary_table_widget.horizontalHeader().height() + 5))
            summary_table_widget.setMaximumHeight(800) # Cap max height, add scroll if needed
            summary_table_widget.resizeRowsToContents() # Will resize down if content is smaller than DEFAULT_TABLE_ROW_HEIGHT

            summary_table_layout.addWidget(summary_table_widget)
            self.abc_cov_inner_plot_layout.addWidget(summary_table_card)


        except Exception as e:
            self.abc_cov_inner_plot_layout.addWidget(QLabel(f"<b>Demand Segmentation Error:</b> {e}"))
            print(f"ABC-COV Analysis Error: {e}")


# --- Analysis Page 3: Data Quality (IMPROVED) ---
class DataQualityPage(BaseEdaPage):
    def __init__(self, app_state):
        super().__init__(app_state)
        self.title_label.setText("Data Quality & Integrity Report")
        self.run_button = QPushButton("Generate Data Quality Report")
        self.run_button.clicked.connect(self.run_analysis)
        self.controls_layout.addWidget(self.run_button)
        self.controls_layout.addStretch()

    def set_data(self, df: pd.DataFrame):
        """Override: This page works on cleaned_df, which we get directly from app_state."""
        self.df = self.app_state.cleaned_df
        if self.df is None: return
        # We still populate these for consistency, though they may not be used
        self.date_col = next((c for c in self.df.columns if pd.api.types.is_datetime64_any_dtype(self.df[c])), None)
        self.numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()

    def run_analysis(self):
        super().run_analysis()
        df = self.app_state.cleaned_df
        if df is None:
            self.plot_layout.addWidget(QLabel("No data available. Please upload a file first."))
            return

        report_layout = QGridLayout()
        report_layout.setSpacing(20)

        summary_card = self._create_summary_card(df)
        missing_values_card = self._create_missing_values_card(df)
        constant_cols_card = self._create_constant_columns_card(df)
        dtype_card = self._create_datatype_summary_card(df)

        report_layout.addWidget(summary_card, 0, 0, 1, 2)
        report_layout.addWidget(missing_values_card, 1, 0)
        
        side_panel_layout = QVBoxLayout()
        side_panel_layout.addWidget(constant_cols_card)
        side_panel_layout.addWidget(dtype_card)
        side_panel_layout.addStretch()
        report_layout.addLayout(side_panel_layout, 1, 1)

        report_layout.setColumnStretch(0, 2)
        report_layout.setColumnStretch(1, 1)

        self.plot_layout.addLayout(report_layout)

    def _create_summary_card(self, df):
        card = QFrame(); card.setObjectName("card")
        layout = QGridLayout(card)
        layout.addWidget(QLabel("<h3>Overall Dataset Health</h3>"), 0, 0, 1, 4)

        num_rows, num_cols = df.shape
        total_missing = df.isnull().sum().sum()
        total_cells = np.prod(df.shape)
        missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0
        duplicate_rows = df.duplicated().sum()
        duplicate_pct = (duplicate_rows / num_rows * 100) if num_rows > 0 else 0

        stats = {
            "Total Rows": f"{num_rows:,}",
            "Total Columns": f"{num_cols:,}",
            "Missing Cells": f"{total_missing:,} ({missing_pct:.2f}%)",
            "Duplicate Rows": f"{duplicate_rows:,} ({duplicate_pct:.2f}%)"
        }
        
        col = 0
        for name, value in stats.items():
            name_label = QLabel(name); name_label.setObjectName("stats_label")
            value_label = QLabel(str(value)); value_label.setObjectName("stats_value_label")
            layout.addWidget(name_label, 1, col)
            layout.addWidget(value_label, 2, col)
            col += 1
            
        return card

    def _create_missing_values_card(self, df):
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.addWidget(QLabel("<h3>Missing Values Analysis</h3>"))

        missing_info = df.isnull().sum().reset_index(name='count').query('count > 0').sort_values(by='count', ascending=False)
        
        if missing_info.empty:
            layout.addWidget(QLabel("✅ No missing values found!"))
            return card

        # The heatmap logic remains the same
        if len(df) < 5000:
            canvas = MatplotlibCanvas(self, height=6)
            # Give the canvas a minimum height to prevent it from collapsing
            canvas.setMinimumHeight(250) 
            ax = canvas.fig.add_subplot(111)
            sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis', ax=ax)
            ax.set_title("Missing Data Pattern")
            canvas.fig.tight_layout()
            layout.addWidget(canvas)
            layout.addWidget(self.create_separator())

        layout.addWidget(QLabel("<b>Columns with Missing Data:</b>"))

        # --- THIS IS THE FIX ---
        # 1. Prepare the DataFrame for display
        display_df = missing_info.copy()
        display_df['percentage'] = (display_df['count'] / len(df) * 100)
        display_df.rename(columns={'index': 'Column', 'count': 'Missing Count', 'percentage': 'Missing %'}, inplace=True)
        # Format the percentage column for better readability
        display_df['Missing %'] = display_df['Missing %'].map('{:.2f}%'.format)

        # 2. Use the single, robust helper function to create the table
        # This function correctly handles all row/column sizing and height limiting.
        table = self.create_table_widget(display_df, max_visible_rows=10)
        # --- END OF FIX ---
        
        layout.addWidget(table)
        
        return card

    def _create_constant_columns_card(self, df):
        card = QFrame(); card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.addWidget(QLabel("<h3>Constant Value Columns</h3>"))

        constant_cols = [col for col in df.columns if df[col].nunique() == 1]
        
        if not constant_cols:
            layout.addWidget(QLabel("✅ No constant value columns found."))
        else:
            list_widget = QListWidget()
            list_widget.addItems(constant_cols)
            layout.addWidget(list_widget)
        
        return card

    def _create_datatype_summary_card(self, df):
        card = QFrame(); card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.addWidget(QLabel("<h3>Column Data Types</h3>"))
        
        dtype_counts = df.dtypes.astype(str).value_counts()
        
        canvas = MatplotlibCanvas(self, height=4)
        ax = canvas.fig.add_subplot(111)
        
        wedges, texts, autotexts = ax.pie(
            dtype_counts.values, 
            labels=dtype_counts.index, 
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette("pastel")
        )
        ax.axis('equal')
        
        canvas.fig.tight_layout()
        layout.addWidget(canvas)
        return card

# --- Analysis Page 4: Product Insights ---
class ProductInsightsPage(BaseEdaPage):
    def __init__(self, app_state):
        super().__init__(app_state)
        self.title_label.setText("Product Lifecycle Insights")
        self.controls_layout.addWidget(QLabel("Product ID Column:")); self.product_combo = QComboBox(); self.controls_layout.addWidget(self.product_combo, 1)
        self.run_button = QPushButton("Generate Product Insights"); self.run_button.clicked.connect(self.run_analysis); self.controls_layout.addWidget(self.run_button)

    def update_controls(self):
        self.product_combo.clear(); self.product_combo.addItems(self.categorical_cols)

    def run_analysis(self):
        super().run_analysis(); df = self.app_state.aggregated_df; product_col = self.product_combo.currentText()
        if df is None or not product_col or self.date_col is None: self.plot_layout.addWidget(QLabel("Please select a Product ID column and ensure a date column is present.")); return
        
        # Ensure target_col is selected and valid
        if not self.numeric_cols:
            self.plot_layout.addWidget(QLabel("No numeric columns available for 'Target Column' in aggregated data.")); return
        target_col = self.numeric_cols[0] # Assuming the first numeric column is the target for now, or add a combo for it.

        try:
            card = QFrame(); card.setObjectName("card"); card_layout = QVBoxLayout(card)
            card_layout.addWidget(QLabel(f"<h3>Product Lifecycle for '{product_col}'</h3>"))

            first_sales = df[df[target_col] > 0].groupby(product_col)[self.date_col].min().reset_index(name='first_sale_date')
            df_plc = pd.merge(df, first_sales, on=product_col, how='left')
            df_plc['product_age_days'] = (df_plc[self.date_col] - df_plc['first_sale_date']).dt.days
            df_plc = df_plc[df_plc['product_age_days'] >= 0].dropna(subset=['product_age_days'])
            
            if df_plc.empty: 
                card_layout.addWidget(QLabel("Could not calculate product age or no positive sales found. Ensure data is correct.")); 
                self.plot_layout.addWidget(card); 
                return
            
            df_plc['product_age_weeks'] = (df_plc['product_age_days'] // 7)
            avg_sales_by_age = df_plc.groupby('product_age_weeks')[target_col].mean().reset_index() # Reset index for seaborn
            
            canvas = MatplotlibCanvas(self, height=6); ax = canvas.fig.add_subplot(111)
            # Corrected: Use ax.grid(True) instead of .set(grid=True)
            sns.lineplot(data=avg_sales_by_age, x='product_age_weeks', y=target_col, ax=ax)
            ax.set_title(f'Average Sales by Product Age (Weeks)')
            ax.grid(True) # Corrected: Use ax.grid(True)
            ax.set_xlabel('Product Age (Weeks)') # Added explicit x-label
            ax.set_ylabel(f'Average {target_col}') # Added explicit y-label

            canvas.fig.tight_layout(); card_layout.addWidget(canvas)
            self.plot_layout.addWidget(card)
        except Exception as e: 
            self.plot_layout.addWidget(QLabel(f"<b>Product Lifecycle Error:</b> {e}"))
            print(f"Product Lifecycle Error: {e}") # Print full error for debugging

# --- Analysis Page 5: Feature Insights ---
class FeatureInsightsPage(BaseEdaPage):
    def __init__(self, app_state):
        super().__init__(app_state)
        self.title_label.setText("Feature Insights")
        self.controls_layout.addWidget(QLabel("Feature Column:")); self.feature_combo = QComboBox(); self.controls_layout.addWidget(self.feature_combo, 1)
        self.controls_layout.addWidget(QLabel("Target Column:")); self.target_combo = QComboBox(); self.controls_layout.addWidget(self.target_combo, 1)
        self.run_button = QPushButton("Generate Feature Insight"); self.run_button.clicked.connect(self.run_analysis); self.controls_layout.addWidget(self.run_button)

    def update_controls(self):
        self.feature_combo.clear(); self.feature_combo.addItems(self.categorical_cols + self.numeric_cols)
        self.target_combo.clear(); self.target_combo.addItems(self.numeric_cols)

    def run_analysis(self):
        super().run_analysis(); df = self.app_state.aggregated_df
        feature_col, target_col = self.feature_combo.currentText(), self.target_combo.currentText()
        if df is None or not feature_col or not target_col or feature_col == target_col: self.plot_layout.addWidget(QLabel("Please select distinct Feature and Target columns.")); return
        
        try:
            card = QFrame(); card.setObjectName("card"); card_layout = QVBoxLayout(card)
            canvas = MatplotlibCanvas(self, height=7); ax = canvas.fig.add_subplot(111)
            if pd.api.types.is_numeric_dtype(df[feature_col]):
                sns.regplot(data=df, x=feature_col, y=target_col, ax=ax, scatter_kws={'alpha':0.4})
                ax.set_title(f"'{target_col}' vs. Numeric Feature '{feature_col}'")
            else:
                top_categories = df[feature_col].value_counts().nlargest(20).index
                sns.boxplot(data=df[df[feature_col].isin(top_categories)], x=feature_col, y=target_col, ax=ax)
                ax.set_title(f"'{target_col}' by Categorical Feature '{feature_col}' (Top 20)")
                ax.tick_params(axis='x', rotation=45, labelsize=9)
            canvas.fig.tight_layout(); card_layout.addWidget(canvas)
            self.plot_layout.addWidget(card)
        except Exception as e: self.plot_layout.addWidget(QLabel(f"<b>Feature Insight Error:</b> {e}"))

# --- Analysis Page 6: Volume Analysis ---
class VolumeAnalysisPage(BaseEdaPage):
    def __init__(self, app_state):
        super().__init__(app_state)
        self.title_label.setText("Volume Analysis")
        
        left_controls_container = QWidget()
        left_layout = QVBoxLayout(left_controls_container)
        left_layout.setContentsMargins(0,0,0,0)
        left_layout.addWidget(QLabel("<b>Group by Dimensions:</b>"))
        self.dim_list = QListWidget(); self.dim_list.setMinimumHeight(120)
        left_layout.addWidget(self.dim_list)
        
        right_controls_container = QWidget()
        right_layout = QVBoxLayout(right_controls_container)
        right_layout.setContentsMargins(0,0,0,0); right_layout.setSpacing(15)
        
        target_layout = QHBoxLayout(); target_layout.addWidget(QLabel("Measure:")); self.target_combo = QComboBox(); target_layout.addWidget(self.target_combo)
        spinbox_layout = QHBoxLayout(); spinbox_layout.addWidget(QLabel("Top/Bottom N:")); self.n_spinbox = QSpinBox(); self.n_spinbox.setRange(1, 100); self.n_spinbox.setValue(5); spinbox_layout.addWidget(self.n_spinbox); spinbox_layout.addStretch()
        self.run_button = QPushButton("Run Volume Analysis"); self.run_button.clicked.connect(self.run_analysis)
        
        right_layout.addLayout(target_layout); right_layout.addLayout(spinbox_layout); right_layout.addStretch(); right_layout.addWidget(self.run_button, 0, Qt.AlignRight)
        
        self.controls_layout.addWidget(left_controls_container, 1)
        self.controls_layout.addWidget(right_controls_container, 1)

    def update_controls(self):
        self.target_combo.clear(); self.target_combo.addItems(self.numeric_cols)
        all_dims = self.grouping_cols + [c for c in self.categorical_cols if c not in self.grouping_cols]
        self.dim_list.clear()
        for col in all_dims:
            item = QListWidgetItem(col); item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if col in self.grouping_cols else Qt.Unchecked)
            self.dim_list.addItem(item)

    def run_analysis(self):
        super().run_analysis()
        df = self.app_state.aggregated_df
        selected_dims = [self.dim_list.item(i).text() for i in range(self.dim_list.count()) if self.dim_list.item(i).checkState() == Qt.Checked]
        target = self.target_combo.currentText(); top_n = self.n_spinbox.value()
        if df is None or not selected_dims or not target: self.plot_layout.addWidget(QLabel("Please select at least one dimension and a measure to analyze.")); return

        try:
            card = QFrame(); card.setObjectName("card"); card_layout = QVBoxLayout(card)
            card_layout.addWidget(QLabel(f"<h3>Volume Analysis by {', '.join(selected_dims)}</h3>"))

            volume_df = df.groupby(selected_dims)[target].sum().sort_values(ascending=False).reset_index()
            table = self.create_volume_table(volume_df, selected_dims, target)
            card_layout.addWidget(table)
            
            card_layout.addWidget(self.create_separator())
            
            graph_layout = QHBoxLayout()
            top_df = volume_df.head(top_n); bottom_df = volume_df.tail(top_n).sort_values(by=target, ascending=True)
            graph_layout.addWidget(self.create_volume_graph(top_df, selected_dims, target, f"Top {len(top_df)} Performers", "viridis"))
            graph_layout.addWidget(self.create_volume_graph(bottom_df, selected_dims, target, f"Bottom {len(bottom_df)} Performers", "rocket_r"))
            card_layout.addLayout(graph_layout)
            self.plot_layout.addWidget(card)
        except Exception as e:
            self.plot_layout.addWidget(QLabel(f"<b>Volume Analysis Error:</b> {e}"))

    def create_volume_table(self, data_df, dimensions, target):
        table = QTableWidget()
        table.setRowCount(len(data_df))
        table.setColumnCount(len(data_df.columns))
        table.setHorizontalHeaderLabels(data_df.columns)
        table.setSortingEnabled(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        for i, row in data_df.iterrows():
            for j, col_name in enumerate(data_df.columns):
                item = QTableWidgetItem(str(row[col_name]))
                if col_name == target:
                    item.setTextAlignment(Qt.AlignRight|Qt.AlignVCenter)
                table.setItem(i, j, item)

        table.resizeColumnsToContents()
        table.setMinimumHeight(300)
        if len(data_df) > 15:
            table.setMaximumHeight(600)
        return table

    def create_volume_graph(self, data_df, dimensions, target, title, palette):
        graph_container = QWidget(); layout = QVBoxLayout(graph_container); layout.addWidget(QLabel(f"<h4>{title}</h4>"))
        data_df = data_df.copy(); data_df['dimension_label'] = data_df[dimensions].astype(str).agg(' | '.join, axis=1)
        canvas = MatplotlibCanvas(self, height=max(4, len(data_df) * 0.6)); ax = canvas.fig.add_subplot(111)
        sns.barplot(data=data_df, x=target, y='dimension_label', ax=ax, palette=palette, hue='dimension_label', legend=False)
        canvas.fig.tight_layout(pad=2.5); canvas.setMinimumHeight(400); layout.addWidget(canvas)
        return graph_container

# --- Analysis Page 7: AutoML ---
class AutoMLPage(BaseEdaPage):
    def __init__(self, app_state):
        super().__init__(app_state)
        self.title_label.setText("Find Best Model with AutoML")
        self.controls_layout.addWidget(QLabel("Segment:")); self.group_combo = QComboBox(); self.controls_layout.addWidget(self.group_combo, 1)
        self.controls_layout.addWidget(QLabel("Target:")); self.target_combo = QComboBox(); self.controls_layout.addWidget(self.target_combo, 1)
        self.controls_layout.addWidget(QLabel("Training Time (sec):")); self.time_limit_spinbox = QSpinBox(); self.time_limit_spinbox.setRange(10, 3600); self.time_limit_spinbox.setValue(60); self.controls_layout.addWidget(self.time_limit_spinbox)
        self.run_button = QPushButton("Find Best Model"); self.run_button.clicked.connect(self.run_analysis); self.controls_layout.addWidget(self.run_button)

    def update_controls(self):
        self.target_combo.clear(); self.target_combo.addItems(self.numeric_cols)
        self.group_combo.clear(); self.group_combo.addItem("Overall")
        if self.grouping_cols: self.group_combo.addItems([" | ".join(map(str, c)) for c in self.df[self.grouping_cols].drop_duplicates().values])

    def run_analysis(self):
        super().run_analysis(); target_col = self.target_combo.currentText()
        if self.df is None or not target_col: self.plot_layout.addWidget(QLabel("No valid data or target selected.")); return
        segment_text = self.group_combo.currentText(); df_to_train = self.df
        if segment_text != "Overall":
            mask = (self.df[self.grouping_cols].astype(str) == segment_text.split(" | ")).all(axis=1)
            df_to_train = self.df[mask]

        self.run_button.setEnabled(False); self.run_button.setText("Training...")
        self.progress_bar = QProgressBar(); self.progress_bar.setRange(0, 0)
        self.plot_layout.addWidget(QLabel("<h3>Training in Progress...</h3>")); self.plot_layout.addWidget(self.progress_bar)

        self.worker = AutoMLWorker(df_to_train, target_col, self.time_limit_spinbox.value(), segment_text.replace(" | ", "_"))
        self.worker.error.connect(self.on_automl_error); self.worker.finished.connect(self.on_automl_finished); self.worker.start()

    def on_automl_error(self, err_msg):
        self.clear_plots(); self.plot_layout.addWidget(QLabel(f"<b>{err_msg}</b>"))
        self.run_button.setEnabled(True); self.run_button.setText("Find Best Model")

    def on_automl_finished(self, leaderboard, logs):
        self.clear_plots(); self.run_button.setEnabled(True); self.run_button.setText("Find Best Model")
        if leaderboard.empty: self.plot_layout.addWidget(QLabel("<h3>Training finished, but no models were successful.</h3>")); return

        best_model = leaderboard.iloc[0]
        summary_card = QFrame(); summary_card.setObjectName("card"); summary_layout = QGridLayout(summary_card)
        summary_layout.addWidget(QLabel("<h3>✅ Training Complete!</h3>"), 0, 0, 1, 2)
        summary_layout.addWidget(QLabel("<b>Best Model Found:</b>"), 1, 0); summary_layout.addWidget(QLabel(f"<b>{best_model['model']}</b>"), 1, 1)
        summary_layout.addWidget(QLabel("<b>Validation Score (RMSE):</b>"), 2, 0); summary_layout.addWidget(QLabel(f"{best_model['score_val']:.4f}"), 2, 1)
        summary_layout.addWidget(QLabel("<b>Total Training Time:</b>"), 3, 0); summary_layout.addWidget(QLabel(f"~{self.time_limit_spinbox.value()} seconds"), 3, 1)
        self.plot_layout.addWidget(summary_card)
        
        results_card = QFrame(); results_card.setObjectName("card"); results_layout = QVBoxLayout(results_card)
        results_tabs = QTabWidget(); results_layout.addWidget(results_tabs); self.plot_layout.addWidget(results_card, 1)
        
        table = QTableWidget()
        table.setRowCount(len(leaderboard))
        table.setColumnCount(len(leaderboard.columns))
        table.setHorizontalHeaderLabels(leaderboard.columns)
        table.setSortingEnabled(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        for i, row in leaderboard.iterrows():
            for j, col in enumerate(leaderboard.columns):
                table.setItem(i, j, QTableWidgetItem(str(row[col])))

        table.resizeColumnsToContents()
        table.setMinimumHeight(400)
        log_viewer = QTextEdit(); log_viewer.setReadOnly(True); log_viewer.setText(logs); log_viewer.setStyleSheet("font-family: 'Monaco', 'Courier New', monospace;")
        results_tabs.addTab(table, "🏆 Leaderboard"); results_tabs.addTab(log_viewer, "📄 Full Logs")


class ParetoAnalysisPage(BaseEdaPage):
    def __init__(self, app_state: AppState):
        super().__init__(app_state)
        self.title_label.setText("Pareto Analysis (80/20 Rule)")

        # Controls for Dimension and Measure
        self.controls_layout.addWidget(QLabel("Dimension:"))
        self.dimension_combo = QComboBox()
        self.controls_layout.addWidget(self.dimension_combo, 1)

        self.controls_layout.addWidget(QLabel("Measure:"))
        self.measure_combo = QComboBox()
        self.controls_layout.addWidget(self.measure_combo, 1)

        self.run_button = QPushButton("Generate Pareto Chart")
        self.run_button.clicked.connect(self.run_analysis)
        self.controls_layout.addWidget(self.run_button)
        self.controls_layout.addStretch() # Push controls to the left

    def update_controls(self):
        """Populates the dimension and measure dropdowns."""
        self.dimension_combo.clear()
        self.measure_combo.clear()

        if self.df is None:
            return

        # Dimensions: Grouping columns + other categorical columns
        all_dims = self.grouping_cols + [c for c in self.categorical_cols if c not in self.grouping_cols]
        if all_dims:
            self.dimension_combo.addItems(sorted(all_dims))

        # Measures: Numeric columns
        if self.numeric_cols:
            self.measure_combo.addItems(sorted(self.numeric_cols))
            # Try to pre-select a 'priority' measure if available
            priority_targets = [col for col in self.numeric_cols if any(k in col.lower() for k in self.PRIORITY_KEYWORDS)]
            if priority_targets:
                self.measure_combo.setCurrentText(priority_targets[0])

    def run_analysis(self):
        """Performs Pareto analysis and displays the results."""
        super().run_analysis() # Clears existing plots

        df = self.app_state.aggregated_df # Pareto is usually on aggregated data
        dimension_col = self.dimension_combo.currentText()
        measure_col = self.measure_combo.currentText()
        
        if df is None or not dimension_col or not measure_col:
            self.plot_layout.addWidget(QLabel("<h3>Please select a Dimension and a Measure to analyze.</h3><p>Ensure data is aggregated first.</p>"))
            return

        try:
            # Group by the selected dimension and sum the measure
            pareto_df = df.groupby(dimension_col)[measure_col].sum().sort_values(ascending=False).reset_index()
            pareto_df.columns = [dimension_col, 'Measure_Value']

            # Calculate cumulative sum and percentage
            pareto_df['Cumulative_Value'] = pareto_df['Measure_Value'].cumsum()
            total_value = pareto_df['Measure_Value'].sum()
            pareto_df['Cumulative_Percentage'] = (pareto_df['Cumulative_Value'] / total_value) * 100

            # Find the 80% cutoff point
            eighty_percent_idx = (pareto_df['Cumulative_Percentage'] >= 80).idxmax()
            vital_few_count = eighty_percent_idx + 1 # 0-indexed

            # Create the Pareto chart
            chart_card = QFrame(); chart_card.setObjectName("card")
            chart_layout = QVBoxLayout(chart_card)
            chart_layout.addWidget(QLabel(f"<h3>Pareto Chart: {measure_col} by {dimension_col}</h3>"))

            canvas = MatplotlibCanvas(self, height=8)
            ax1 = canvas.fig.subplots()
            ax2 = ax1.twinx() # Create a second y-axis

            # Bar plot for individual measure values
            sns.barplot(x=dimension_col, y='Measure_Value', data=pareto_df, ax=ax1, palette='Blues_d', hue=dimension_col, legend=False)
            ax1.set_xlabel(dimension_col)
            ax1.set_ylabel(measure_col)
            ax1.tick_params(axis='x', rotation=45)

            # Line plot for cumulative percentage
            ax2.plot(pareto_df[dimension_col], pareto_df['Cumulative_Percentage'], color='red', marker='o', linestyle='-', linewidth=2)
            ax2.set_ylabel('Cumulative Percentage (%)')
            ax2.set_ylim(0, 100)
            ax2.grid(True, linestyle='--', alpha=0.6)

            # Add 80% line and label for the vital few
            ax2.axhline(80, color='grey', linestyle=':', linewidth=1.5)
            # CORRECTED LINE: Use ax2.transAxes for reliable positioning relative to the plot area
            ax2.text(0.98, 80, '80%', va='center', ha='right', backgroundcolor='w', transform=ax2.transData)
            ax1.axvline(x=vital_few_count - 0.5, color='green', linestyle='--', linewidth=1.5, label=f'Vital Few ({vital_few_count} {dimension_col})')
            ax1.legend(loc='upper left')

            canvas.fig.tight_layout()
            canvas.setMinimumHeight(550)
            chart_layout.addWidget(canvas)
            self.plot_layout.addWidget(chart_card)

            self.plot_layout.addWidget(self.create_separator())

            # Create the summary table
            summary_card = QFrame(); summary_card.setObjectName("card")
            summary_layout = QVBoxLayout(summary_card)
            summary_layout.addWidget(QLabel(f"<h3>Pareto Summary Table</h3>"))

            summary_table = QTableWidget()
            summary_table.setRowCount(len(pareto_df))
            summary_table.setColumnCount(len(pareto_df.columns))
            summary_table.setHorizontalHeaderLabels(pareto_df.columns)
            summary_table.setSortingEnabled(True)
            # Set fixed row height
            summary_table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
            summary_table.verticalHeader().setDefaultSectionSize(DEFAULT_TABLE_ROW_HEIGHT)


            for r_idx, row_data in pareto_df.iterrows():
                for c_idx, value in enumerate(row_data):
                    item = QTableWidgetItem()
                    if pareto_df.columns[c_idx] in ['Measure_Value', 'Cumulative_Value']:
                        item.setText(f"{value:,.2f}")
                        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    elif pareto_df.columns[c_idx] == 'Cumulative_Percentage':
                        item.setText(f"{value:.2f}%")
                        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    else:
                        item.setText(str(value))
                        item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                    summary_table.setItem(r_idx, c_idx, item)

            summary_table.resizeColumnsToContents()
            for i in range(summary_table.columnCount()):
                summary_table.horizontalHeader().setSectionResizeMode(i, QHeaderView.Stretch)
            summary_table.setMinimumHeight(max(400, summary_table.verticalHeader().length() + summary_table.horizontalHeader().height() + 5))
            summary_table.resizeRowsToContents() # Will resize down if content is smaller than DEFAULT_TABLE_ROW_HEIGHT

            summary_layout.addWidget(summary_table)
            self.plot_layout.addWidget(summary_card)



        except Exception as e:
            self.plot_layout.addWidget(QLabel(f"<b>Pareto Analysis Error:</b> {e}"))
