# ui/ai_analyst_tab.py

import os
import sys
import pandas as pd
import matplotlib
import seaborn as sns
import base64
from io import BytesIO, StringIO
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QEvent
from typing import Dict, List, Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QFrame, QLabel,
    QHBoxLayout, QLineEdit, QTextEdit, QMessageBox,
    QScrollArea, QApplication, QSizePolicy, QComboBox
)
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langchain_perplexity import ChatPerplexity
from tools.data_analysis_tools import (
      DescribeData,
      CorrelationAnalysis,
      VisualizeData,
      CalculateWMAPE
)

# Use non-interactive backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from typing import TypedDict
import re
import warnings
warnings.filterwarnings('ignore')

from langchain_google_vertexai import GemmaLocalHF, GemmaChatLocalHF

import ast

# New imports for LangChain agent usage
try:
    from langchain.chat_models import ChatOpenAI
    from langchain.agents import initialize_agent, AgentType
    from langchain.agents import create_react_agent, AgentExecutor
    from langchain import hub
    from langchain.tools import Tool as LangTool
    from langchain.prompts import ChatPromptTemplate
except Exception:
    # If LangChain agent classes are not available, the code will raise at runtime when trying to use them.
    ChatOpenAI = None
    initialize_agent = None
    AgentType = None
    LangTool = None

# === Global DataFrame Holder ===
class ToolState:
    df = None
    _execution_globals = {}

# === Enhanced Code Executor with Full Library Support ===
class EnhancedCodeExecutor:
    def __init__(self):
        self.name = "execute_code"
        self.description = "Executes Python code for data analysis with full library support"
        self._setup_safe_environment()

    def _setup_safe_environment(self):
        """Setup a safe but comprehensive execution environment"""
        import math
        import statistics
        import itertools
        import collections
        import functools
        import operator
        import random
        import datetime
        import json
        
        # Start with basic libraries that are always available
        self.available_libraries = {
            'math': math,
            'statistics': statistics,
            'itertools': itertools,
            'collections': collections,
            'functools': functools,
            'operator': operator,
            'random': random,
            're': re,
            'datetime': datetime,
            'json': json,
            'os': os,
            'sys': sys
        }
        
        # Add data science libraries
        try:
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns

            
            self.available_libraries.update({
                'numpy': np, 'np': np,
                'pandas': pd, 'pd': pd,
                'matplotlib': matplotlib,
                'plt': plt, 'pyplot': plt,
                'seaborn': sns, 'sns': sns
            })
            
            # Try additional libraries one by one
            optional_libs = [
                ('scipy', 'scipy'),
                ('sklearn', 'sklearn'), 
                ('plotly', 'plotly'),
                ('bokeh', 'bokeh'),
                ('altair', 'altair'),
                ('statsmodels', 'statsmodels'),
                ('networkx', 'networkx'),
                ('requests', 'requests')
            ]
            
            for lib_name, import_name in optional_libs:
                try:
                    lib = __import__(import_name)
                    self.available_libraries[lib_name] = lib
                except ImportError:
                    pass  # Skip if not available
                    
        except ImportError as e:
            # Keep basic libraries even if pandas/numpy fail
            pass

    def execute(self, code: str) -> str:
        """Executes code in a comprehensive but safe environment."""
        if ToolState.df is None:
            return "Error: No data available. Please select a dataset first."

        try:
            # Basic security checks - only block truly dangerous operations
            dangerous_patterns = [
                r'__import__\s*\(',
                r'eval\s*\(',
                r'exec\s*\(',
                r'compile\s*\(',
                r'globals\s*\(\)',
                r'locals\s*\(\)',
                r'getattr\s*\(',
                r'setattr\s*\(',
                r'delattr\s*\(',
                r'hasattr\s*\(',
                r'subprocess',
                r'os\.system',
                r'os\.popen',
                r'os\.spawn',
                r'os\.fork',
                r'os\.exec',
                r'open\s*\([^)]*["\'][rwa]'  # file operations
            ]
            
            code_check = code.lower()
            for pattern in dangerous_patterns:
                if re.search(pattern, code_check):
                    return f"SecurityError: Potentially unsafe operation detected: {pattern}"

            # Setup execution environment
            execution_globals = self.available_libraries.copy()
            execution_globals.update(ToolState._execution_globals)
            
            execution_locals = {
                "df": ToolState.df.copy(),  # Work with a copy
                "__name__": "__main__"
            }

            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            # Clear any existing plots
            plt.close('all')

            try:
                # Execute the code
                exec(code, execution_globals, execution_locals)
                
                # Update global state with any new variables (except built-ins)
                for key, value in execution_locals.items():
                    if not key.startswith('__') and key != 'df':
                        ToolState._execution_globals[key] = value
                
            finally:
                # Restore stdout
                printed_output = captured_output.getvalue().strip()
                sys.stdout = old_stdout

            # Handle plot generation
            plot_html = ""
            if plt.get_fignums():
                try:
                    buf = BytesIO()
                    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', 
                              facecolor='white', edgecolor='none')
                    buf.seek(0)
                    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
                    plot_html = f"![Plot](data:image/png;base64,{img_b64})"
                    plt.close('all')
                except Exception as plot_error:
                    plot_html = f"Plot generation error: {str(plot_error)}"
                    plt.close('all')

            # Build comprehensive response
            response_parts = []
            
            if printed_output:
                response_parts.append(f"**Output:**\n```\n{printed_output}\n```")
            
            if plot_html:
                response_parts.append(f"**Visualization:**\n{plot_html}")
            
            # Check if any significant results were produced
            if not response_parts:
                # Check if code created any variables or modifications
                new_vars = [k for k in execution_locals.keys() 
                           if k not in ['df', '__name__'] and not k.startswith('__')]
                if new_vars:
                    response_parts.append(f"**Code executed successfully.** Created variables: {', '.join(new_vars[:5])}")
                else:
                    response_parts.append("**Code executed successfully** (no output produced)")

            return "\n\n".join(response_parts)

        except SyntaxError as e:
            plt.close('all')
            return f"**Syntax Error:** Line {e.lineno}: {e.msg}\n\nPlease check your code syntax."
        
        except ImportError as e:
            plt.close('all')
            missing_lib = str(e).split("'")[1] if "'" in str(e) else "unknown"
            return f"**Import Error:** Library '{missing_lib}' not available.\n\nTry using: pandas, numpy, matplotlib, seaborn, scipy, sklearn"
        
        except Exception as e:
            plt.close('all')
            error_type = type(e).__name__
            return f"**{error_type}:** {str(e)}\n\nPlease check your code and try again."

# === Enhanced LangGraph State ===
class AgentState(TypedDict):
    messages: List[BaseMessage]
    iteration: int
    short_term_memory: List[str]
    long_term_insights: List[str]
    current_dataset: str
    execution_context: Dict

# === Enhanced Worker Thread ===
class AgentWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, df: pd.DataFrame, query: str, 
                 chat_history: List[BaseMessage], 
                #  api_key = "xy", 
                 dataset_name: str = "Dataset"):
        super().__init__()
        self.df = df.copy()
        # self.api_key = api_key
        self.query = query
        self.chat_history = chat_history
        self.dataset_name = dataset_name
        self.long_term_insights = []

    def _extract_code(self, content: str) -> str:
        """Extract Python code from a model response (can contain ``` blocks)."""
        patterns = [
            r'```python\s*(.*?)\s*```',
            r'```py\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```'
        ]
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            if matches:
                return matches[0].strip()
        # fallback: collect lines that look like code
        lines = content.splitlines()
        code_lines = []
        for line in lines:
            if line.strip().startswith(('import ', 'from ', 'df.', 'plt.', 'sns.', 'print(', 'for ', 'if ', 'def ', '@')):
                code_lines.append(line)
        return '\n'.join(code_lines).strip()

    def _extract_tool_call(self, content: str):
        """Return (tool_name, args_dict) or (None, None)"""
        m = re.search(r'<tool>\s*(.*?)\s*</tool>\s*<args>\s*(.*?)\s*</args>', content, re.DOTALL | re.IGNORECASE)
        if not m:
            return None, None
        tool_name = m.group(1).strip()
        args_text = m.group(2).strip()
        try:
            args = ast.literal_eval(args_text) if args_text else {}
        except Exception:
            args = None
        return tool_name, args

    def _ask_model(self, model) -> str:
        """Invoke the model and return text (safe wrapper).

        Prefer LangChain agent.run(prompt) when model is an agent created via initialize_agent,
        otherwise fall back to model.invoke([...]) if available.
        """
        try:
            # If this "model" is a LangChain agent-like object that exposes .run(), prefer that.
            if hasattr(model, "run"):
                # agent.run may expect a single string and return a string
                return model.run(self.query)
            # Fallback for objects exposing invoke (e.g., ChatPerplexity)
            if hasattr(model, "invoke"):
                resp = model.invoke([HumanMessage(content=self.query)])
                content = getattr(resp, "content", str(resp))
                return content
            # Generic fallback: try calling as function
            try:
                resp = model(self.query)
                return getattr(resp, "content", str(resp))
            except Exception:
                return str(resp)
        except Exception as e:
            return f"ModelError: {str(e)}"

    def run(self):
        try:
            self.progress.emit("Initializing root agent (LangChain)...")
            print("AgentWorker: starting with dataset:", self.dataset_name)

            # Build local tool instances (existing local tool classes)
            tools = [
                DescribeData(),
                CorrelationAnalysis(),
                VisualizeData(),
                CalculateWMAPE()
            ]

            def find_tool(name: str):
                return next((t for t in tools if getattr(t, "name", t.__class__.__name__) == name), None)

            # Prepare dataset summary for prompt
            numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            dataset_summary = f"DATASET: {self.dataset_name} ({self.df.shape[0]} rows × {self.df.shape[1]} columns)\nNumeric cols: {numeric_cols}\nCategorical cols: {categorical_cols}\nSample:\n{self.df.head(3).to_string()}\n"

            # Build the decision prompt
            system_prompt = f"""
You are a data analysis agent. For the user query below you must choose ONE of the following options (and return only that):

1) Call a tool using EXACT markup (no extra text):
   <tool>tool_name</tool><args>{{"arg1": "value1", ...}}</args>

   Available tools:
   - describe_data: Get statistical description of the data
   - correlation_analysis: Correlation matrix and heatmap
   - visualize_data: Create plots (histogram, scatter, box, bar)
   - calculate_WMAPE: Calculate WMAPE grouped by intersection

OR

2) Return a Python code block (```python ... ```) that uses the dataframe variable `df` and common libraries (pandas, numpy, matplotlib, seaborn) to perform analysis and produce textual or visual output.

Do NOT mix tool markup and code. The dataset summary is below.

{dataset_summary}

USER QUESTION: "{self.query}"

Respond with EITHER a single tool call markup OR a single python code block.
"""
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),  
            ])
            self.progress.emit("Asking root agent for plan...")

            # --- Create a LangChain agent (preferred) ------------------------------------------------
            agent = None
            try:
                
                # Create an LLM instance (uses ChatOpenAI). If an API key is required by ChatOpenAI,
                # user should configure OPENAI_API_KEY in the environment. We set from self.api_key when present.
                # if self.api_key:
                #     os.environ.setdefault("OPENAI_API_KEY", self.api_key)
                
                                
                llm = GemmaLocalHF(
                    model_name="google/gemma-3-270m",
                    # hf_access_token=self.api_key,
                    hf_access_token="your token"
                )
                # Convert local tool instances into LangChain Tool wrappers
                langchain_tools = []
                for t in tools:
                    tname = getattr(t, "name", t.__class__.__name__)
                    tdesc = getattr(t, "description", "")
                    # wrap tool call into a simple function that accepts kwargs
                    def make_fn(tool_obj):
                        def _fn(text: str = ""):
                            # Many of our tools expect structured args; this wrapper keeps backward compatibility.
                            # If text is a JSON/dict-like string, try to parse to dict via literal_eval.
                            try:
                                parsed = ast.literal_eval(text) if text else {}
                            except Exception:
                                parsed = {"text": text}
                            try:
                                if isinstance(parsed, dict):
                                    return tool_obj._run(**parsed)
                                if isinstance(parsed, (list, tuple)):
                                    return tool_obj._run(*parsed)
                                return tool_obj._run(parsed)
                            except TypeError:
                                # fallback: try passing raw text
                                return tool_obj._run(text)
                        return _fn
                    langchain_tools.append(LangTool(name=tname, func=make_fn(t), description=tdesc))

                # Initialize agent with bound tools
                # agent = initialize_agent(tools = langchain_tools, llm = llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)

                # # Ask agent for a decision using agent.run(...) (preferable)
                # decision = self._ask_model(agent, prompt)
                
                agent = create_react_agent(llm, langchain_tools, prompt = prompt)
                agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
                decision = self._ask_model(agent_executor, prompt)
            except Exception as e:
                # If agent creation fails, raise a helpful error (ChatPerplexity root agent removed as requested)
                raise RuntimeError(f"Failed to create LangChain agent: {str(e)}")

            print("Root agent decision:\n", decision)

            ToolState.df = self.df  # set dataset for any tool/code

            code_executor = EnhancedCodeExecutor()

            # --- Tool-call handling (unchanged) ------------------------------------------------------
            tool_name, tool_args = self._extract_tool_call(decision)
            if tool_name:
                print("TOOL CALL DETECTED:", tool_name, tool_args)
                self.progress.emit(f"Executing tool: {tool_name} ...")
                tool_obj = find_tool(tool_name)
                if tool_obj is None:
                    raise RuntimeError(f"Tool not found: {tool_name}")

                # Execute the tool
                try:
                    tool_result = tool_obj._run(**(tool_args or {}))
                except TypeError:
                    tool_result = tool_obj._run(*(tool_args if isinstance(tool_args, (list, tuple)) else []))
                except Exception as e:
                    raise RuntimeError(f"Tool execution error: {str(e)}")

                print("TOOL RESULT (truncated):", str(tool_result)[:1000])
                # If the tool returned a plot encoded as markdown image, keep it.
                # Otherwise, run a code-generation step to postprocess `tool_result`.
                postproc_needed = True
                if isinstance(tool_result, str) and tool_result.strip().startswith("![Plot]"):
                    postproc_needed = True

                final_output = ""
                if postproc_needed:
                    try:
                        if hasattr(tool_result, "copy"):
                            ToolState._execution_globals['tool_result'] = tool_result.copy()
                        else:
                            ToolState._execution_globals['tool_result'] = tool_result

                        post_prompt = f"""
You are a Python code generator. The user asked: "{self.query}"
A tool named "{tool_name}" produced a variable available as `tool_result`.
Write only Python code (no surrounding text) that uses `tool_result` and available libraries (pandas, numpy, matplotlib, seaborn) to produce a useful final output (print summaries, produce plots as matplotlib that the executor can capture). 
Do not write file IO. Keep code robust to empty data. Return only the code block (use ```python ... ```).
"""
                        self.progress.emit("Generating postprocessing code...")
                        post_response = self._ask_model(agent, post_prompt)
                        generated_code = self._extract_code(post_response)
                        print("GENERATED POSTPROCESSING CODE:\n", generated_code)
                        if not generated_code:
                            if isinstance(tool_result, str):
                                final_output = str(tool_result)
                            elif hasattr(tool_result, "head"):
                                final_output = "Tool returned a DataFrame. Showing head:\n" + tool_result.head(5).to_string()
                            else:
                                final_output = str(tool_result)
                        else:
                            self.progress.emit("Executing generated postprocessing code...")
                            result = code_executor.execute(generated_code)
                            print("POSTPROCESS EXECUTE RESULT:\n", result)
                            if any(x in result for x in ["Error:", "Exception:", "Traceback", "SecurityError", "Import Error:"]):
                                correction_prompt = f"""
The following Python code failed when executed. Provide ONLY the corrected Python code (no explanations).

ORIGINAL CODE:
```python
{generated_code}
```

ERROR:
{result}

The variable `tool_result` is available in the execution environment. Use the same available libraries.
"""
                                self.progress.emit("Requesting corrected code...")
                                correction_resp = self._ask_model(agent, correction_prompt)
                                corrected_code = self._extract_code(correction_resp)
                                print("CORRECTED CODE:\n", corrected_code)
                                if corrected_code:
                                    result2 = code_executor.execute(corrected_code)
                                    print("RE-EXECUTE RESULT:\n", result2)
                                    if any(x in result2 for x in ["Error:", "Exception:", "Traceback", "SecurityError", "Import Error:"]):
                                        raise RuntimeError("Postprocessing failed after correction:\n" + result2)
                                    else:
                                        final_output = result2
                                else:
                                    raise RuntimeError("Model failed to provide corrected code.")
                            else:
                                final_output = result
                    finally:
                        ToolState._execution_globals.pop('tool_result', None)

                combined = ""
                if isinstance(tool_result, str):
                    combined += str(tool_result)
                else:
                    try:
                        if hasattr(tool_result, "head"):
                            combined += f"Tool '{tool_name}' returned DataFrame: {getattr(tool_result, 'shape', '')}\n" + tool_result.head(5).to_string()
                        else:
                            combined += str(tool_result)
                    except Exception:
                        combined += str(tool_result)

                if final_output:
                    combined += "\n\n--- Postprocessing ---\n" + final_output

                self.finished.emit(str(combined))
                return

            # --- Code path (unchanged) ---------------------------------------------------------------
            code_candidate = self._extract_code(decision)
            if not code_candidate:
                code_prompt = f"""
No tool markup found. The user asked: "{self.query}".
Write a self-contained Python code block (```python ... ```) that uses the dataframe `df` and available libraries (pandas, numpy, matplotlib, seaborn) to produce textual/visual analysis results. Do not use file IO. Return only the code.
"""
                self.progress.emit("Requesting analysis code from agent...")
                code_resp = self._ask_model(agent, code_prompt)
                code_candidate = self._extract_code(code_resp)
                print("GENERATED ANALYSIS CODE:\n", code_candidate)

            if not code_candidate:
                raise RuntimeError("Agent did not return executable code or tool call.")

            self.progress.emit("Executing analysis code...")
            print("EXECUTING CODE:\n", code_candidate)
            exec_result = code_executor.execute(code_candidate)
            print("EXEC RESULT:\n", exec_result)

            if any(x in exec_result for x in ["Error:", "Exception:", "Traceback", "SecurityError", "Import Error:"]):
                correction_prompt = f"""
The following Python code (for dataset analysis) failed when executed. Provide ONLY the corrected Python code (no explanations).

ORIGINAL CODE:
```python
{code_candidate}
```

ERROR:
{exec_result}

Use the dataframe `df` and available libraries. Keep behavior aligned with the user's request: "{self.query}"
"""
                self.progress.emit("Requesting corrected analysis code...")
                corr_resp = self._ask_model(agent, correction_prompt)
                corrected_code = self._extract_code(corr_resp)
                print("CORRECTED ANALYSIS CODE:\n", corrected_code)
                if corrected_code:
                    final_result = code_executor.execute(corrected_code)
                    print("FINAL EXEC RESULT:\n", final_result)
                    if any(x in final_result for x in ["Error:", "Exception:", "Traceback", "SecurityError", "Import Error:"]):
                        raise RuntimeError("Analysis failed after correction:\n" + final_result)
                else:
                    raise RuntimeError("Agent failed to provide corrected analysis code.")
            else:
                final_result = exec_result

            self.finished.emit(str(final_result))

        except Exception as e:
            err_msg = f"AgentWorker error: {str(e)}"
            print(err_msg)
            self.error.emit(err_msg)
        finally:
            ToolState.df = None
            plt.close('all')

# === Enhanced AI Analyst Tab ===
class AIAnalystTab(QWidget):
    def __init__(self, app_state):
        super().__init__()
        self.app_state = app_state
        # self.api_key = "your token"
        self.worker = None
        self.chat_history = []
        self.long_term_insights = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(12)

        # Enhanced Header
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                stop:0 #3498db, stop:1 #2c3e50);
            border-radius: 12px;
            padding: 20px;
            color: white;
        """)
        header_layout = QVBoxLayout(header_frame)
        
        title = QLabel("🧠 AI Data Analyst")
        title.setStyleSheet("font-size: 26px; font-weight: bold; color: white; margin: 0;")
        
        subtitle = QLabel("Advanced data analysis with full Python library support. Ask complex questions and get insights with visualizations.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("font-size: 14px; color: #ecf0f1; margin: 5px 0 0 0;")
        
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        layout.addWidget(header_frame)

        # Dataset Selection with Status
        selection_frame = QFrame()
        selection_frame.setStyleSheet("""
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 10px;
            padding: 15px;
        """)
        selection_layout = QHBoxLayout(selection_frame)

        df_label = QLabel("📊 Active Dataset:")
        df_label.setStyleSheet("font-weight: bold; color: #2c3e50; font-size: 14px;")

        self.df_selector = QComboBox()
        self.df_selector.setStyleSheet("""
            QComboBox {
                padding: 8px 12px;
                font-size: 14px;
                border: 1px solid #bdc3c7;
                border-radius: 6px;
                background: white;
                min-width: 200px;
            }
            QComboBox:focus { border-color: #3498db; }
            QComboBox::drop-down { border: none; width: 30px; }
            QComboBox::down-arrow {
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #7f8c8d;
            }
        """)

        self.dataset_info = QLabel()
        self.dataset_info.setStyleSheet("color: #7f8c8d; font-size: 12px;")

        selection_layout.addWidget(df_label)
        selection_layout.addWidget(self.df_selector)
        selection_layout.addWidget(self.dataset_info)
        selection_layout.addStretch()

        layout.addWidget(selection_frame)

        # Chat Area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("""
            QScrollArea { 
                border: 1px solid #e9ecef; 
                background: white;
                border-radius: 10px;
            }
            QScrollBar:vertical {
                width: 12px;
                background: #f8f9fa;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #adb5bd;
                border-radius: 6px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: #6c757d;
            }
        """)

        self.chat_widget = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_widget)
        self.chat_layout.setAlignment(Qt.AlignTop)
        self.chat_layout.setSpacing(15)
        self.chat_layout.setContentsMargins(20, 20, 20, 20)

        # Add welcome message
        self.add_message("""
**Welcome to AI Data Analyst! 🎯**

I can help you analyze your data using the full power of Python. Here are some example questions:

• *"What are the key trends in this data?"*
• *"Show me correlations between variables"*
• *"Find outliers and unusual patterns"*
• *"Create a comprehensive visualization dashboard"*
• *"Perform statistical analysis and hypothesis testing"*

Select your dataset above and ask me anything!
        """.strip(), is_user=False)

        self.scroll.setWidget(self.chat_widget)
        layout.addWidget(self.scroll, 1)

        # Enhanced Input Area
        input_frame = QFrame()
        input_frame.setStyleSheet("""
            background: #f8f9fa;
            border-top: 1px solid #e9ecef;
            padding: 15px;
            border-radius: 0 0 10px 10px;
        """)
        input_layout = QVBoxLayout(input_frame)

        # Quick actions
        quick_actions = QHBoxLayout()
        quick_buttons = [
            ("📈 Overview", "Give me a comprehensive overview of this dataset"),
            ("🔍 Patterns", "Find interesting patterns and correlations"),
            ("📊 Visualize", "Create informative visualizations"),
            ("🧮 Statistics", "Perform statistical analysis")
        ]

        for btn_text, query in quick_buttons:
            btn = QPushButton(btn_text)
            btn.setStyleSheet("""
                QPushButton {
                    background: #e9ecef;
                    border: 1px solid #dee2e6;
                    padding: 6px 10px;
                    border-radius: 15px;
                    font-size: 11px;
                    color: #495057;
                }
                QPushButton:hover {
                    background: #3498db;
                    color: white;
                    border-color: #3498db;
                }
            """)
            btn.clicked.connect(lambda checked, q=query: self._set_query(q))
            quick_actions.addWidget(btn)
        
        quick_actions.addStretch()
        input_layout.addLayout(quick_actions)

        # Main input
        main_input = QHBoxLayout()
        
        self.input = QLineEdit()
        self.input.setPlaceholderText("Ask me anything about your data... (supports complex analysis requests)")
        self.input.setStyleSheet("""
            QLineEdit {
                padding: 14px 16px;
                font-size: 15px;
                border: 2px solid #dee2e6;
                border-radius: 25px;
                background: white;
                selection-background-color: #3498db;
            }
            QLineEdit:focus {
                border-color: #3498db;
                background: white;
            }
        """)

        self.btn = QPushButton("🚀 Analyze")
        self.btn.setFixedSize(100, 50)
        self.btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3498db, stop:1 #2980b9);
                color: white;
                border: none;
                border-radius: 25px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2980b9, stop:1 #1f618d);
            }
            QPushButton:disabled {
                background: #bdc3c7;
                color: #7f8c8d;
            }
        """)

        main_input.addWidget(self.input)
        main_input.addWidget(self.btn)
        input_layout.addLayout(main_input)

        layout.addWidget(input_frame)

        # Progress indicator
        self.progress_label = QLabel()
        self.progress_label.setStyleSheet("color: #3498db; font-style: italic; margin: 5px 0;")
        self.progress_label.hide()
        input_layout.addWidget(self.progress_label)

        # Connect events
        self.btn.clicked.connect(self.run_analysis)
        self.input.returnPressed.connect(self.run_analysis)
        self.df_selector.currentTextChanged.connect(self.update_dataset_info)

        # Connect app state signals
        self.app_state.data_loaded.connect(self.update_df_selector)
        self.app_state.data_cleaned.connect(self.update_df_selector)
        self.app_state.data_aggregated.connect(self.update_df_selector)

        # Initial setup
        self.update_df_selector()

    def _set_query(self, query: str):
        """Set predefined query"""
        self.input.setText(query)
        self.input.setFocus()

    def update_df_selector(self):
        """Update dropdown with available DataFrames"""
        self.df_selector.blockSignals(True)
        current = self.df_selector.currentText()
        self.df_selector.clear()

        sources = []
        if self.app_state.raw_df is not None and not self.app_state.raw_df.empty:
            sources.append("Raw Data")
        if self.app_state.cleaned_df is not None and not self.app_state.cleaned_df.empty:
            sources.append("Cleaned Data")
        if self.app_state.aggregated_df is not None and not self.app_state.aggregated_df.empty:
            sources.append("Aggregated Data")

        if not sources:
            sources.append("No Data Available")

        self.df_selector.addItems(sources)

        # Restore selection or set default
        if current in sources:
            self.df_selector.setCurrentText(current)
        elif len(sources) > 1 and sources[0] != "No Data Available":
            self.df_selector.setCurrentIndex(0)

        self.df_selector.blockSignals(False)
        self.update_dataset_info()

    def update_dataset_info(self):
        """Update dataset information display"""
        df = self.get_selected_dataframe()
        if df is not None and not df.empty:
            info = f"{df.shape[0]} rows × {df.shape[1]} columns"
            self.dataset_info.setText(info)
            self.dataset_info.setStyleSheet("color: #27ae60; font-size: 12px;")
        else:
            self.dataset_info.setText("No data selected")
            self.dataset_info.setStyleSheet("color: #e74c3c; font-size: 12px;")

    def get_selected_dataframe(self) -> Optional[pd.DataFrame]:
        """Get the currently selected DataFrame"""
        source_name = self.df_selector.currentText()
        if not source_name or source_name == "No Data Available":
            return None

        df_map = {
            "Raw Data": self.app_state.raw_df,
            "Cleaned Data": self.app_state.cleaned_df,
            "Aggregated Data": self.app_state.aggregated_df
        }
        return df_map.get(source_name)

    def resizeEvent(self, event):
        """Handle window resize events"""
        super().resizeEvent(event)
        QTimer.singleShot(100, self._update_message_widths)

    def _update_message_widths(self):
        """Update all message widths for responsive design"""
        available_width = int(self.scroll.viewport().width() * 0.85)
        
        for i in range(self.chat_layout.count()):
            item = self.chat_layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                # Find all QTextEdit widgets in the message
                for text_widget in widget.findChildren(QTextEdit):
                    text_widget.setMaximumWidth(available_width)
                    text_widget.document().setTextWidth(available_width - 40)
                    
                    # Recalculate height
                    doc_height = text_widget.document().size().height()
                    text_widget.setFixedHeight(int(doc_height + 25))

    def add_message(self, text: str, is_user: bool = False):
        """Add a message to the chat with enhanced formatting"""
        # Convert markdown to HTML
        html_text = self._markdown_to_html(text)

        # Create message container
        message_container = QWidget()
        container_layout = QHBoxLayout(message_container)
        container_layout.setContentsMargins(0, 8, 0, 8)

        # Create text widget
        text_widget = QTextEdit()
        text_widget.setReadOnly(True)
        text_widget.setHtml(html_text)
        text_widget.setLineWrapMode(QTextEdit.WidgetWidth)
        
        # Calculate initial dimensions
        available_width = int(self.scroll.viewport().width() * 0.75)
        text_widget.setMaximumWidth(available_width)
        text_widget.document().setTextWidth(available_width - 40)
        
        doc_height = text_widget.document().size().height()
        text_widget.setFixedHeight(int(doc_height + 25))

        # Enhanced styling
        if is_user:
            text_widget.setStyleSheet("""
                QTextEdit {
                    font-family: 'Segoe UI', Arial, sans-serif;
                    font-size: 14px;
                    line-height: 1.5;
                    padding: 16px;
                    border-radius: 18px;
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #3498db, stop:1 #2980b9);
                    color: white;
                    border: none;
                    selection-background-color: rgba(255,255,255,0.3);
                }
            """)
            container_layout.addStretch()
            container_layout.addWidget(text_widget)
        else:
            text_widget.setStyleSheet("""
                QTextEdit {
                    font-family: 'Segoe UI', Arial, sans-serif;
                    font-size: 14px;
                    line-height: 1.6;
                    padding: 16px;
                    border-radius: 18px;
                    background: #f8f9fa;
                    color: #2c3e50;
                    border: 1px solid #e9ecef;
                    selection-background-color: #3498db;
                }
            """)
            container_layout.addWidget(text_widget)
            container_layout.addStretch()

        self.chat_layout.addWidget(message_container)
        
        # Auto-scroll to bottom
        QTimer.singleShot(50, self._scroll_to_bottom)

    def _scroll_to_bottom(self):
        """Smooth scroll to bottom"""
        scrollbar = self.scroll.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _markdown_to_html(self, text: str) -> str:
        """Enhanced markdown to HTML conversion"""
        # Code blocks
        text = re.sub(
            r'```(\w+)?\n?(.*?)\n?```',
            r'<div style="margin: 12px 0;"><pre style="background: #2c3e50; color: #ecf0f1; padding: 16px; border-radius: 8px; overflow-x: auto; font-family: Consolas, Monaco, monospace; font-size: 13px; line-height: 1.4;"><code>\2</code></pre></div>',
            text, flags=re.DOTALL
        )
        
        # Inline code
        text = re.sub(r'`([^`]+)`', r'<code style="background: #f1f3f4; padding: 2px 6px; border-radius: 3px; font-family: monospace; font-size: 13px;">\1</code>', text)
        
        # Bold and italic
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        
        # Images (plots)
        text = re.sub(
            r'!\[Plot\]\(data:image/png;base64,(.*?)\)',
            r'<div style="text-align: center; margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 12px;"><img src="data:image/png;base64,\1" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 16px rgba(0,0,0,0.1);"></div>',
            text
        )
        
        # Lists
        text = re.sub(r'^• (.+)', r'<li style="margin: 4px 0;">\1</li>', text, flags=re.MULTILINE)
        text = re.sub(r'(<li.*?</li>)', r'<ul style="margin: 8px 0; padding-left: 20px;">\1</ul>', text)
        
        # Line breaks
        text = text.replace('\n', '<br>')
        
        return text

    def run_analysis(self):
        """Run comprehensive data analysis"""
        query = self.input.text().strip()
        if not query:
            return

        # Validate dataset
        df = self.get_selected_dataframe()
        if df is None or df.empty:
            QMessageBox.warning(
                self, 
                "No Dataset Selected", 
                "Please load data and select a dataset from the dropdown above."
            )
            return

        selected_name = self.df_selector.currentText()
        
        # Add user message
        self.add_message(f"**{query}**", is_user=True)
        
        # Clear input and disable controls
        self.input.clear()
        self.btn.setEnabled(False)
        self.input.setEnabled(False)
        
        # Show progress
        self.progress_label.setText("🔄 Initializing analysis...")
        self.progress_label.show()

        # Start analysis
        self.worker = AgentWorker(df, 
                                #   self.api_key, 
                                  query, self.chat_history, selected_name)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.progress.connect(self.on_progress_update)
        self.worker.start()

        # Update chat history
        self.chat_history.append(HumanMessage(content=query))

    def on_progress_update(self, status: str):
        """Update progress indicator"""
        self.progress_label.setText(f"🔄 {status}")

    def on_analysis_finished(self, response: str):
        """Handle successful analysis completion"""
        self.progress_label.hide()
        
        # Add AI response
        self.add_message(response)
        
        # Update chat history
        self.chat_history.append(AIMessage(content=response))
        
        # Keep chat history manageable
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-15:]
        
        # Re-enable controls
        self.btn.setEnabled(True)
        self.input.setEnabled(True)
        self.input.setFocus()

    def on_analysis_error(self, error: str):
        """Handle analysis errors"""
        self.progress_label.hide()
        
        error_message = f"""
**❌ Analysis Error**

{error}

**Suggestions:**
• Check if your dataset has the required columns
• Try rephrasing your question
• Ensure the dataset contains appropriate data types
• Contact support if the issue persists
        """.strip()
        
        self.add_message(error_message)
        
        # Re-enable controls
        self.btn.setEnabled(True)
        self.input.setEnabled(True)
        self.input.setFocus()

    def clear_chat(self):
        """Clear chat history and messages"""
        # Clear UI
        for i in reversed(range(self.chat_layout.count())):
            item = self.chat_layout.takeAt(i)
            if item and item.widget():
                item.widget().deleteLater()
        
        # Clear memory
        self.chat_history.clear()
        self.long_term_insights.clear()
        ToolState._execution_globals.clear()
        
        # Add welcome message back
        self.add_message("""
**Chat cleared! 🧹**

Ready for new analysis. Select your dataset and ask me anything!
        """.strip(), is_user=False)

    def closeEvent(self, event):
        """Clean up on close"""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait(3000)  # Wait up to 3 seconds
        
        # Clear global state
        ToolState.df = None
        ToolState._execution_globals.clear()
        plt.close('all')
        
        super().closeEvent(event)


