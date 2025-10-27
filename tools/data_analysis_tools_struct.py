from langchain_core.tools import BaseTool
from typing import Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO
import base64
import ast

# Try importing the @tool decorator from langchain, with a safe fallback.
try:
    from langchain.tools import tool
except Exception:
    try:
        from langchain_core.tools import tool
    except Exception:
        # Minimal fallback: attach metadata and return the function unchanged.
        def tool(name=None, description=None):
            def decorator(func):
                func.name = name or getattr(func, "__name__", None)
                func.description = description
                return func
            return decorator

@tool
def describe_data(column: str = None) -> str:
    """
    Get statistical description of the dataframe or a specific column.
    Args:
        column (str, optional): Column name to describe. If None, describe all numeric columns.
    Returns:
        str: String representation of the description or an error message if no dataframe is available.
    """
    from ui.ai_analyst_tab import ToolState
    df = ToolState.df
    if df is None:
        return "No dataframe available"
    if column and column in df.columns:
        return df[column].describe().to_string()
    return df.describe().to_string()


@tool
def correlation_analysis(**kwargs) -> str:
    """
    Compute correlation matrix for numeric columns and return a heatmap image as a markdown data URL.
    Args:
        figsize (tuple, optional): Figure size for the heatmap (default (10, 8)).
        annot (bool, optional): Whether to annotate cells (default True).
        cmap (str, optional): Colormap to use (default 'coolwarm').
        title (str, optional): Title for the plot (default 'Correlation Matrix').
    Returns:
        str: Markdown data URL of the generated heatmap PNG, or an error message.
    """
    from ui.ai_analyst_tab import ToolState
    df = ToolState.df
    if df is None:
        return "No dataframe available"

    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) < 2:
        return "Not enough numeric columns for correlation analysis"

    corr = df[numeric_cols].corr()
    plt.figure(figsize=kwargs.get("figsize", (10, 8)))
    sns.heatmap(corr, annot=kwargs.get("annot", True), cmap=kwargs.get("cmap", "coolwarm"), center=0)
    plt.title(kwargs.get("title", "Correlation Matrix"))

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return f"![Plot](data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')})"


@tool
def visualize_data(plot_type: str = None, x_col: str = None, y_col: str = None, **kwargs) -> str:
    """
    Create visualizations: histogram, scatter, box, bar. Returns a PNG image as a markdown data URL.
    Args:
        plot_type (str): Type of plot. One of 'histogram', 'scatter', 'box', 'bar'.
        x_col (str): Column name for the x-axis (required).
        y_col (str, optional): Column name for the y-axis (required for scatter/box/bar).
        figsize (tuple, optional): Figure size (default (10, 6)).
        Any additional kwargs are forwarded to the seaborn plotting function.
    Returns:
        str: Markdown data URL of the generated plot PNG, or an error message.
    """
    from ui.ai_analyst_tab import ToolState
    df = ToolState.df
    if df is None:
        return "No dataframe available"
    if plot_type is None or x_col is None:
        return "plot_type and x_col required"

    plt.figure(figsize=kwargs.get("figsize", (10, 6)))

    try:
        if plot_type == "histogram":
            sns.histplot(data=df, x=x_col, **{k: v for k, v in kwargs.items()})
        elif plot_type == "scatter" and y_col:
            sns.scatterplot(data=df, x=x_col, y=y_col, **{k: v for k, v in kwargs.items()})
        elif plot_type == "box":
            sns.boxplot(data=df, x=x_col, y=y_col, **{k: v for k, v in kwargs.items()})
        elif plot_type == "bar":
            sns.barplot(data=df, x=x_col, y=y_col, **{k: v for k, v in kwargs.items()})
        else:
            return "Unsupported plot type or missing y_col for this plot"
    except Exception as e:
        plt.close()
        return f"Plotting error: {str(e)}"

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return f"![Plot](data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')})"


@tool
def calculate_WMAPE(df_id, forecast_col: str = None, actuals_col: str = None, **kwargs):
    """
    Calculate WMAPE grouped by intersection or combination.
    Args:
        df (pandas.DataFrame): The input dataframe containing forecast and actuals.
        forecast_col (str): The name of the forecast column.
        actuals_col (str): The name of the actuals column.
    Returns:
        pandas.Series or str: Grouped WMAPE values per 'intersection', or an error message.
    """
    # from ui.ai_analyst_tab import ToolState
    # df = pd.read_csv(r"C:\Users\Preet Lodaya\MLOps_Bot_temp\Results\Stat_forecast_20120426.csv") #ToolState.df
    df = df_store.get(df_id)
    if df is None:
        return "No dataframe available"
    if forecast_col is None or actuals_col is None:
        return "forecast_col and actuals_col required"

    try:
        grouped = df.groupby('intersection').apply(
            lambda x: np.sum(np.abs(x[forecast_col] - x[actuals_col])) / np.sum(x[actuals_col]) * 100
        )
        description = f"WMAPE calculated between {forecast_col} and {actuals_col} for dataframe with id {df_id}"
        if kwargs['agg'] is not None:
            agg_levels = [v for k,v in kwargs['agg'].items()]
            grouped_id = df_store.put(grouped, id = f"Accuracy_{'_'.join(agg_levels)}")
            description += f", aggregated by {', '.join(agg_levels)}"
        else:
            grouped_id = df_store.put(grouped, id = f"Accuracy_{df_id}")
        # return grouped
        return json.dumps({
            "status": "ok",
            "df_id": grouped_id,
            "description": description
        })

    except Exception as e:
        return f"WMAPE calculation error: {str(e)}"


# --- Helper: wrap existing tool instances into LangChain Tool-compatible objects ---
def wrap_tool_for_langchain(tool_obj):
    """
    Produce a LangChain Tool wrapper from the provided tool instance.
    The returned tool expects a single string input that can be a Python literal (dict/list/tuple)
    or raw text. It will call the tool_obj._run(...) appropriately.
    """
    try:
        try:
            from langchain.tools import Tool
        except Exception:
            from langchain_core.tools import Tool  # fallback
    except Exception:
        return None

    def wrapper(input_str: str = ""):
        try:
            parsed = ast.literal_eval(input_str) if input_str else {}
        except Exception:
            parsed = input_str

        try:
            if isinstance(parsed, dict):
                return tool_obj._run(**parsed)
            elif isinstance(parsed, (list, tuple)):
                return tool_obj._run(*parsed)
            else:
                return tool_obj._run(parsed)
        except TypeError:
            return tool_obj._run(input_str)
        except Exception as e:
            return f"Tool execution error: {str(e)}"

    # Prefer Tool.from_function if available
    try:
        return Tool.from_function(wrapper, name=getattr(tool_obj, "name", None), description=getattr(tool_obj, "description", None))
    except Exception:
        try:
            return Tool(func=wrapper, name=getattr(tool_obj, "name", None), description=getattr(tool_obj, "description", None))
        except Exception:
            return None
