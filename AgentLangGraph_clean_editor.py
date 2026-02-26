# ============================================================================
# MODULE-LEVEL CONSTANTS
# ============================================================================

MAX_ITERATIONS = 5
MAX_ANALYST_ITERATIONS = 10
EXECUTOR_TIMEOUT = 9
CODE_OUTPUT_LIMIT = 2000
PLANNER_LLM = "gpt-5"
EXECUTOR_LLM = "gpt-5-mini"
PROGRAMMER_LLM = "gpt-4.1-mini"

# PLANNER_LLM = "gpt-5-mini"
# EXECUTOR_LLM = "gpt-5-mini"
# PROGRAMMER_LLM = "gpt-4.1-mini"

# ============================================================================
# IMPORTS AND DEPENDENCIES
# ============================================================================

import code
import json
import os
import subprocess
import tempfile
import uuid
import ast
import logging
from typing import Any, Dict, List, Optional, Annotated, Literal
from dataclasses import dataclass, field
from datetime import datetime
from langgraph.prebuilt import InjectedState
import pandas as pd
import numpy as np
import multiprocessing as mp
import traceback
# ... other imports ...


	
	# Run agent...
# ============================================================================
# LOGGING CONFIGURATION - SUPPRESS VERBOSE OUTPUT
# ============================================================================

# Suppress verbose logging from libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("langchain_core").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("langchain_openai").setLevel(logging.WARNING)

# Optional: Set root logger to WARNING to catch all unfiltered logs
logging.getLogger().setLevel(logging.WARNING)

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.types import StreamWriter
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.types import Command
from typing_extensions import Annotated, List, Literal, Union, TypedDict, runtime
import re, operator


# try to import hub (some langchain builds don't include it); provide tiny fallback
try:
	from langchain import hub
except Exception:
	class _HubFallback:
		@staticmethod
		def pull(name: str) -> str:
			if name == "hwchase17/react":
				# Minimal ReAct-style prompt used by executor; formatted with .format(...)
				return (
					"You are an agent using tools. Available tool names: {tool_names}\n\n"
					"Tools:\n{tools}\n\n"
					
					"When thinking, follow ReAct format. Use these tokens exactly:\n"
					"Thought: <text>\n"
					"Action: <tool_name>\n"
					"Action Input: <json or text>\n"
					"Observation: <result from tool>\n"
					"Final Answer: <output if done>\n\n"
					"Context (scratchpad):\n{agent_scratchpad}\n"
					"You will receive input in the following format: id :description \n\n"
					"Input:\n{input}\n\n"
				)
			raise ImportError(f"hub.pull fallback has no entry for '{name}'")
	hub = _HubFallback()


import operator
# from typing import Annotated, List, Literal, Union, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from CodeExecutor_test import run_generated_code_in_subprocess
from dotenv import load_dotenv
from openai import AsyncOpenAI

# This looks for a .env file in the current directory
load_dotenv()

# Now the client will automatically find the key from the environment
client = AsyncOpenAI()
# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

intersection_level = []
previous_attempts: Dict[str, Any] = {}  # Track failed attempts for error context

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def dict_merge(old: Dict, new: Dict) -> Dict:
	"""Merge two dicts, with new keys overriding old ones"""
	return {**old, **new}

def df_summary(df: pd.DataFrame) -> str:
	"""Generate summary of DataFrame"""
	dataset_summary = f"DATASET: Results ({df.shape[0]} rows × {df.shape[1]} columns) Columns Names: {', '.join(df.columns.tolist())}\n\n"
	return dataset_summary

def track_attempt(attempt_id: str, goal: str, error: str, code: str) -> None:
	"""Track failed attempts for debugging and error context"""
	global previous_attempts
	previous_attempts[attempt_id] = {
		"timestamp": datetime.now().isoformat(),
		"goal": goal,
		"error": error,
		"code": code,
		"traceback": error
	}

def get_previous_attempt_context(attempt_id: str) -> Optional[str]:
	"""Retrieve error context from previous failed attempt"""
	global previous_attempts
	if attempt_id in previous_attempts:
		attempt = previous_attempts[attempt_id]
		return f"Previous error:\n{attempt['error']}\n\nPrevious code:\n{attempt['code']}"
	return None

def clear_previous_attempts() -> None:
	"""Clear attempt history"""
	global previous_attempts
	previous_attempts = {}

def _tool_name(fn) -> str:
	return getattr(fn, "__name__", getattr(fn, "name", fn.__class__.__name__))

# ============================================================================
# TYPE DEFINITIONS
# ============================================================================


class DataFrameInfo(TypedDict):
	"""Schema for individual DataFrame store entries"""
	# DataFrame: pd.DataFrame
	Summary: str
	Description: Optional[str]

class GlobalState(TypedDict):
	"""Shared state between planner and executor agents. Now includes DataFrame store."""
	messages: Annotated[List[BaseMessage], operator.add]
	available_tools: List[str]

	# Planner outputs 
	task_lists: Annotated[List[List[str]], operator.add]

	DataAnalyst_reports: Annotated[List[List[str]], operator.add]
	next_step: Literal["DataAnalyst", "GraphAnalyst", "FINISH", ""]
	dataframe_info: Dict[str, DataFrameInfo]

	# Control flow
	iteration_count: int
	query_response: str

class WorkerState(TypedDict):
	# 'add_messages' ensures we append history, not overwrite it
	messages: Annotated[List[BaseMessage], operator.add]
	current_task: str
	dataframe_info: Annotated[Dict[str, DataFrameInfo], dict_merge]
	generated_code: str 
	coding_error: str #Annotated[Dict[str,str], dict_merge]
	coding_error_traceback: str
	iteration_count: int

# class AnalysisRequest(BaseModel):
# 	"""Call this when you need more data or analysis."""
# 	reasoning: str = Field(..., description = "Your thought process. Why are we taking this step? How will it help? Explain briefly.")
# 	task: List[str] = Field(..., description = "List of distinct and numbered tasks to be performed/insights to be generated by the next agent")
# 	# next_step: Literal["DataAnalyst", "FINISH"] = Field(..., description="Next agent to be triggered or FINISH (if you have all required info)")
# 	next_step: Literal["DataAnalyst"] = "DataAnalyst"
	
# class FinalResponse(BaseModel):
# 	"""Call this ONLY when you have sufficient information to answer."""
# 	reasoning: str = Field(..., description="Why the analysis is complete.")
# 	answer: str = Field(..., description="The final comprehensive answer.")
# 	next_step: Literal["FINISH"] = "FINISH"

# class SupervisorDecision(BaseModel):
#     """Supervisor's decision: either request more analysis or provide final response."""
#     decision: Union[AnalysisRequest, FinalResponse] = Field(
#         ..., 
#         discriminator="next_step",
#         description="Either AnalysisRequest (next_step='DataAnalyst') or FinalResponse (next_step='FINISH')"
#     )

class SupervisorDecision(BaseModel):
	"""Call this when you need more data or analysis."""
	reasoning: str = Field(..., description = "Your thought process. Why are we taking this step? How will it help? Explain briefly.")
	next_step: Literal["DataAnalyst", "FINISH"] = Field(
		..., 
		description="Select 'DataAnalyst' to request work or 'FINISH' to provide the final result.")
	# task: Optional[List[str]] = Field(..., description = """List of distinct and numbered tasks to be performed/insights to be 
	#                                   generated by the next agent. Eg:- ['1. Find top 5 products by sales', '2. Calculate monthly growth rate']""")  # REQUIRED if next_step is 'DataAnalyst'
	task: Optional[str] = Field(..., description = """Task or insight to be extracted from the data by the Analyst""")  # REQUIRED if next_step is 'DataAnalyst'
	# next_step: Literal["DataAnalyst", "FINISH"] = Field(..., description="Next agent to be triggered or FINISH (if you have all required info)")
	answer: Optional[str] = Field(
		None, 
		description="REQUIRED if decision_type is 'FINISH'. The final comprehensive answer."
	)

# ============================================================================
# TOOLS
# ============================================================================

# @tool
# def get_ts_frequency_fn(state: AgentState,tool_input: str) -> str:
# 	"""Determines the time series frequency of the date column using the mode statistic of time differences.
# 	Args: JSON string with keys:
# 		df_id (str): The ID of the DataFrame to use
# 		date_col (str): The date column name
# 	"""
# 	try:
# 		payload = json.loads(tool_input) if isinstance(tool_input, str) else tool_input
# 		df_id = payload["df_id"]
# 		date_col = payload["date_col"]
		
# 		df = state['dataframe_store'].get(df_id)['DataFrame']
# 		if date_col not in df.columns:
# 			return json.dumps({"status": "error", "description": f"Date column '{date_col}' not found"})
		
# 		df[date_col] = pd.to_datetime(df[date_col])
# 		df = df.sort_values(by=date_col)
# 		date_diffs = df[date_col].diff().dropna()
# 		most_common_diff = date_diffs.mode()[0]
		
# 		if most_common_diff == pd.Timedelta(days=1):
# 			frequency = 'Daily'
# 		elif most_common_diff == pd.Timedelta(weeks=1):
# 			frequency = 'Weekly'
# 		elif pd.Timedelta(days=28) <= most_common_diff <= pd.Timedelta(days=31):
# 			frequency = 'Monthly'
# 		else:
# 			frequency = f'Irregular (most common diff: {most_common_diff})'
		
# 		return json.dumps({"status": "ok", "frequency": frequency})
# 	except Exception as e:
# 		return json.dumps({"status": "error", "description": str(e)})

@tool
def get_state_variable_fn(tool_input) -> str:
	"""Retrieves and returns the value of any state variable for inspection.
	Args: JSON string with keys:
		variable_name (str): Name of the state variable to retrieve (e.g., 'messages', 'requirement_results', 'is_resolved')
	Returns: JSON string with the variable value or error
	"""
	try:
		payload = json.loads(tool_input) if isinstance(tool_input, str) else tool_input
		var_name = payload.get("variable_name")
		if not var_name:
			return json.dumps({"status": "error", "description": "Missing 'variable_name'"})
		
		return json.dumps({
			"status": "ok",
			"note": f"State variable '{var_name}' requested. Call this tool during requirement execution to inspect state."
		})
	except Exception as e:
		return json.dumps({"status": "error", "description": str(e)})

def get_dependency_versions() -> Dict[str, str]:
	"""Get versions of key dependencies."""
	versions = {}
	try:
		import pandas as pd
		versions['pandas'] = pd.__version__
	except:
		versions['pandas'] = 'unknown'
	
	try:
		import numpy as np
		versions['numpy'] = np.__version__
	except:
		versions['numpy'] = 'unknown'
	
	return versions

def _tool_name(fn) -> str:
	return getattr(fn, "__name__", getattr(fn, "name", fn.__class__.__name__))

class MultiAgentAnalysisSystem:
	"""High-level interface for multi-agent analysis"""
	
	def __init__(self, intersection_level: List[str], objective: str, dataframes: Optional[Dict[str, Dict[str, Any]]] = None,
              supervisor_llm: str = PLANNER_LLM, executor_llm: str = EXECUTOR_LLM, programmer_llm: str = PROGRAMMER_LLM):
		"""
		Args:
			dataframes: Dict mapping df_id to dict with keys:
				- 'DataFrame': pd.DataFrame (required)
				- 'Description': str (optional)
				Example: {
					'sales_data': {
						'DataFrame': df1,
						'Description': 'Historical sales records'
					},
					'forecast_results': {
						'DataFrame': df2,
						'Description': 'Model forecast outputs'
					}
				}
			model: LLM model to use
		"""
		self.input_dataframes = dataframes
		self.df_store = {key:value['DataFrame'] for key,value in self.input_dataframes.items()} if len(self.input_dataframes) > 0 else {}
		self.supervisor_llm = supervisor_llm
		self.executor_llm = executor_llm
		self.programmer_llm = programmer_llm
		self.data_analyst_tools = self._setup_data_analyst_tools()
		self.graph = self._build_graph()
		self.result_store = {}
		self.intersection_level = intersection_level
		self.objective = objective

	def make_dataframe_info(self) -> Dict[str, DataFrameInfo]:
		"""Generate dataframe_info dict from df_store"""
		dataframe_info: Dict[str, DataFrameInfo] = {}
		for df_id, df_info in self.input_dataframes.items():
			dataframe_info[df_id] = {
				"Summary": df_summary(df_info['DataFrame']),
				"Description": df_info.get("Description",f"Input DataFrame: {df_id}")}
		return dataframe_info
	
	def _setup_data_analyst_tools(self):
		"""Setup tools with ToolRuntime for state access."""

		@tool
		def list_dfs_tool(runtime: ToolRuntime = None) -> Command:
			"""From the df_store, returns list of available DataFrame ids, the shape and columns of each DataFrame and descriptions (what does the data represent)."""
			return self.list_dfs_fn(runtime)

		@tool
		def calculate_accuracy_fn(
			forecast_cols: List[str],
			actuals_col: str,
			date_col: str,
			df_id: str,
			error_calculation_level: Optional[List[str]] = None,
			error_aggregation_level: Optional[List[str]] = None,
			runtime: ToolRuntime = None  # ✅ Injected, hidden from LLM
		) -> Command:
			"""Calculates Accuracy of generated forecasts using Weighted MAPE (WMAPE) metric.
			Expects both actual and prediction values to be present in same dataframe.
			Stores the following:-
				1. bias/Error (Forecast - Actual) (Summed at aggregation level)
				2. absolute error/AE (|Forecast - Actual|) (Summed at aggregation level)
				3. Accuracy Percentage i.e. 100*(1 - WAPE)
				4. Actuals
			for each unique error_aggregation_level combination.

			Args:
				forecast_cols: List of forecast columns. Metrics will be calculated for all columns.
				actuals_col: Actual values column name
				date_col: Date column name
				df_id: The ID of the DataFrame to use
				error_calculation_level: Level for error calculation (default: intersection_level + date_col)
				error_aggregation_level: Level for aggregation (default: intersection_level)
	
			USAGE INSTRUCTIONS:-
			1) Use default value of error_calculation_level unless specified in explicit terms by user. 
				For eg.:- a. Query:- 'Intersection with best accuracy? calculate error at State - SKU level'. Implies -> error_calculation_level=['State','SKU']
						b. Query:- 'Find 5 Worst performing Continent - Style comb.'. Implies -> error_aggregation_level = ['Continent','Style'] but default error_calculation_level
			2) Give all the forecast columns you are interested in calculating above metrics for in one go.
			"""
	
			try:

				# Set defaults
				if error_calculation_level is None:
					error_calculation_level = self.intersection_level + [date_col]
				if error_aggregation_level is None:
					error_aggregation_level = self.intersection_level

				df = self.df_store[df_id]

				if isinstance(forecast_cols, str):
					forecast_cols = [forecast_cols]
				
				if not all(col in df.columns for col in forecast_cols) or actuals_col not in df.columns:
					print("Required columns not found")
					return Command(update={
						"messages": [ToolMessage(
							content="Required columns not found",
							tool_call_id=runtime.tool_call_id
						)]
					})
				
				abs_error_cols = [f'Absolute_Error_{f}' for f in forecast_cols]
				bias_cols = [f'Bias_{f}' for f in forecast_cols]
				accuracy_cols = [f'Accuracy_{f}' for f in forecast_cols]
				
				grouped_df = df.groupby(
					list(set(error_aggregation_level).union(set(error_calculation_level))), 
					as_index=False
				)[forecast_cols + [actuals_col]].sum()
				
				grouped_df[bias_cols] = grouped_df[forecast_cols].subtract(grouped_df[actuals_col], axis=0)
				grouped_df[abs_error_cols] = grouped_df[bias_cols].abs()
				
				grouped_acc = grouped_df.groupby(error_aggregation_level, as_index=False).apply(
					lambda x: 100 - (x[abs_error_cols].sum() * 100 / x[actuals_col].sum())
				)
				grouped_acc.columns = error_aggregation_level + accuracy_cols
				grouped_errors = grouped_df.groupby(error_aggregation_level, as_index=False)[
					bias_cols + abs_error_cols + [actuals_col]
				].sum()
				grouped = pd.merge(grouped_acc, grouped_errors, on=error_aggregation_level, how='inner')
				
				result_id = f"Accuracy_{'_'.join(forecast_cols)}_{'_'.join(error_aggregation_level)}"
				description = (
					f"Accuracy numbers with errors calculated at "
					f"{'-'.join(error_calculation_level)} and aggregated at "
					f"{'-'.join(error_aggregation_level)} "
					f"{'for forecasts: ' + ', '.join(forecast_cols)}"
				)
				
				self.df_store[result_id] = grouped.copy()
				print(description + f" and stored as dataframe with id: {result_id}")
				return Command(update={
					"messages": [ToolMessage(
						content=f"{description} and stored as dataframe with id: {result_id}",
						tool_call_id=runtime.tool_call_id
					)],
					"dataframe_info": {
						result_id: {
							"Summary": df_summary(grouped),
							"Description": description
						}
					}
				})
				
			except Exception as e:
				print(f"Error in calculate_accuracy: {str(e)}")
				return Command(update={
					"messages": [ToolMessage(
						content=f"Error in calculate_accuracy: {str(e)}",
						tool_call_id=runtime.tool_call_id
					)]
				})

		@tool
		def code_run_tool_fn(
			df_ids: List[str],
			insight: Optional[str] = None,
			store_df: Optional[Dict[str,str]] = {},
			instructions: Optional[str] = None,
			timeout: Optional[int] = EXECUTOR_TIMEOUT,
			runtime: ToolRuntime = None  # ✅ Injected, hidden from LLM
		) -> Command:
			"""Generates and Exectes Python code to operate on DataFrames.
			
			Args:
				df_ids: The IDs of the DataFrames to be used
				insight (Optional): Explicit insight required from data. Think of exact values/conclusions 	
    			store_df (Optional): A dictionary of format {new_df_id : description} with IDs and descriptions of the dataframes (both intermediate and final) to be stored in df_store 
				instructions (Optional): Additional remarks/steps/instructions to be incorporated in code generation 
				timeout: Timeout in seconds (Optional. Avoid or give conservative number)
			
			USAGE INSTRUCTIONS:-
			1) This tool CANNOT be parallelized. DONOT call code_run_tool_fn call multiple times at once.
   			2) At least one out of of insight or store_df arguements must be provided.
			3) Always be mindful of the available dataframes and their summaries before framing Goal for code generation.
			4) Has no memory of previous execution.
			"""
			# 1) Avoid overwritting input dataframes with ids:- {list(self.input_dataframes.keys())}.
						
			# return self.code_generator_fn(
			# 	goal=goal,
			# 	df_ids=df_ids,
			# 	store_df=store_df,
			# 	runtime=runtime
			# )
			try:
				print("Available dataframes:- ",runtime.state.get("dataframe_info").keys())
				df_ids = df_ids or []
				gen_messages = []
				generated_code = runtime.state.get("generated_code")

				# 1) Generate code if a goal is provided (code_generator_fn now returns a dict)
				if (not store_df) and (not insight):
					return Command(update={"messages" : [ToolMessage(content="Neither insight nor store_df was passed. Hence, no objective of code generation")]})
				else:
					gen_result = self.code_generator_fn(df_ids=df_ids, insight=insight, 
                                        				store_df=store_df, instructions=instructions, runtime=runtime)
					# gen_result is a dict like {"messages": [...], "generated_code": "...", "coding_error": ""}
					gen_messages = gen_result.get("messages", "")
					generated_code = gen_result.get("generated_code", "")
					
					if len(generated_code) == 0:
						# Return a Command indicating generation failure
						msgs = [ToolMessage(content=gen_messages if gen_messages else "Code generation failed or returned no code.", 
                        					tool_call_id=getattr(runtime, "tool_call_id", None))]
						return Command(update={"messages": msgs, "generated_code":generated_code})
					
				# 2) Execute the (newly generated or pre-existing) code (code_executor_fn now returns a dict)
				exec_result = self.code_executor_fn(runtime=runtime, df_ids=df_ids, gen_code=generated_code, timeout=timeout)
				exec_messages = exec_result.get("messages", "")

				# 3) Merge generation + execution outputs into one update
				merged_messages = ""
				if gen_messages:
					merged_messages += f"Code Generator messages: {gen_messages}\n"
				if exec_messages:
					merged_messages += f"Code Executor messages: {exec_messages}\n"
				
				merged_update: Dict[str, Any] = {}
				if merged_messages:
					merged_update["messages"] = [ToolMessage(content=merged_messages, tool_call_id=runtime.tool_call_id)]
				if exec_result.get("dataframe_info"):
					merged_update["dataframe_info"] = exec_result.get("dataframe_info")
				# propagate coding error info
				if exec_result.get("coding_error") is not None:
					merged_update["coding_error"] = exec_result.get("coding_error")
					merged_update["coding_error_traceback"] = exec_result.get("coding_error_traceback", "")
				# surface generated_code for visibility
				if generated_code:
					merged_update["generated_code"] = generated_code

				return Command(update=merged_update)

			except Exception as e:
				tb = traceback.format_exc()
				return Command(update={
					"messages": [ToolMessage(content=f"Error in code_run_tool_fn: {str(e)}", tool_call_id=getattr(runtime, "tool_call_id", None))],
					"coding_error": str(e),
					"coding_error_traceback": tb
				})

		# Return tools: list_dfs_tool, calculate_accuracy_fn, combined code_run_tool_fn
		return [list_dfs_tool, calculate_accuracy_fn, code_run_tool_fn]

	def list_dfs_fn(self, runtime: ToolRuntime) -> Command:
		"""From the df_store, returns list of available DataFrame ids, their summaries and descriptions."""
		
		df_info = runtime.state.get("dataframe_info")
		if df_info is None:
			return Command(update={
				"messages": [ToolMessage(
					content="No dataframe_info in state",
					tool_call_id=runtime.tool_call_id
				)]
			})
		
		print("Available dataframes:", df_info.keys())
		return Command(update={
			"messages": [ToolMessage(
				content=json.dumps(df_info, indent=2),
				tool_call_id=runtime.tool_call_id
			)]
		})

	def code_generator_fn(
		self,
		df_ids: List[str],
		insight: str,
  		store_df: Dict,
		instructions: str,
		runtime: ToolRuntime
	) -> Dict[str, Any]:
	
		"""Generate code using LLM."""
		print(f"\n📝 CODE GENERATOR INVOKED")
		
		
		state = runtime.state
		dep_versions = get_dependency_versions()
		# code_prompt = """
		# You are a Python coding assistant. Produce ONLY Python code (no explanation).

		# You will be provided required DataFrames through a single variable:

		# 	dfs : Dict[str, pandas.DataFrame]

		# Keys are DataFrame IDs (strings). Values are pandas DataFrames.
		# ALWAYS access DataFrames using:
		# 	df = dfs["<df_id>"]

		# Your Goal: {goal}

		# Available DataFrames' IDs: {available_ids}
		# Summary of available DataFrames (ID -> Summary): {input_df_summary}
		# Must generate output as DataFrame: {store_output_df}

		# OUTPUT CONTRACT:
		# 1. The final output must always be stored in a variable named 'result'.
		# 2. If store_output_df is False:
		# 	- result must be a string that directly answers the goal.
		# 	- Any intermediate calculations must be converted to clear text in result.
		# 	- NEVER print or return dataframes entirely; summarize or extract insights and include in result. 
		# 3. If store_output_df is True:
		# 	- result must be a dict with EXACTLY these keys:
		# 		a. DataFrame (pd.DataFrame): The output DataFrame
		# 		b. df_id (str, optional): Contextual id for the resulting DataFrame
		# 		c. Description (str, optional): Description of what the contained data represents
		# 	- No other formats allowed when store_output_df is True.
		# 4. Include metric values in final output where applicable.

		# INSTRUCTIONS:-
		# 	1. Never assume any variable named 'df', 'df_<id>', or others.
		# 	2. NEVER import anything except pandas as pd and numpy as np (already provided).
		# """

		code_prompt = """
		You are a Python coding assistant. Produce ONLY Python code (no explanation).

		You will be provided required DataFrames through a single variable:
			dfs : Dict[str, pandas.DataFrame]
		Keys are DataFrame IDs (strings). Values are pandas DataFrames.
		ALWAYS access DataFrames using: df = dfs["<df_id>"]

		Available DataFrames' IDs: {available_ids}
		Summary of available DataFrames (ID -> Summary): {input_df_summary}

		OUTPUT CONTRACT:
		The final output must ALWAYS be stored in a variable named 'result'.
		'result' MUST be a dictionary with EXACTLY the following structure:
		
		result = {{
			"response": " ... your text answer/insight here ... ",
			"dataframes": {{
				"new_df_id_1": {{ "DataFrame": df_obj_1, "Description": "desc_1" }},
				"new_df_id_2": {{ "DataFrame": df_obj_2, "Description": "desc_2" }}
			}}
		}}

		YOUR GOAL (Populate the result as mentioned):-
		'Insight' :- {insight}, 
		'DataFrames to Store' - {store_df},

		GUIDELINES:
		
		1. 'response' (str): Specifically answering the required 'Insight'.
		2. 'dataframes' (dict): A nested dictionary of 'DataFrames to Store'. 
		   - Keys are unique IDs (Contextual id) for the new dataframes.
		   - Values are dicts containing the 'DataFrame' object and a 'Description'(Description of what the contained data represents).
		   - If no dataframes need to be stored, leave this empty: "dataframes": {{}}
		3. Generate code being mindful of the following instructions - 
  				{instructions}

		INSTRUCTIONS:-
			1. Never assume any variable named 'df', 'df_<id>', or others.
			2. NEVER import anything except pandas as pd and numpy as np.
		"""
		
		code_prompt += f""" Never generate a new dataframe with same ID as input dataframes':- {list(self.input_dataframes.keys())}.

		ENVIRONMENT INFO:
		- pandas version: {dep_versions.get('pandas', 'unknown')}
		- numpy version: {dep_versions.get('numpy', 'unknown')}
		- Python 3.8+"""

		df_info = state.get("dataframe_info", {})
		summaries = {}
		missing = []
		
		for df_id in df_ids:
			try:
				summaries[df_id] = df_info.get(df_id, {}).get('Summary', '')
			except Exception:
				missing.append(df_id)
		
		if missing:
			print(f"df_ids not found: {missing}")
			return {
				"messages": [ToolMessage(
					content=f"df_ids not found: {missing}",
					tool_call_id=runtime.tool_call_id
				)]
			}
		input_df_summary = "\n".join([f"{df_id} : {summaries[df_id]}" for df_id in df_ids])

		# Include error context from previous attempt if available
		if state.get("coding_error"):
			
			code_prompt += (
				f"\n\nThe previous attempt for the code: \n"
    			f"{state.get('generated_code')}\n\n"
       			f"failed with the following error:\n"
				f"{state.get('coding_error')}\n"
				f"Traceback: {state.get('coding_error_traceback')}\n"
				f"Please fix the bug and regenerate the code defensively."
			)
		_template_vars = ["insight", "available_ids", "input_df_summary", "store_df","instructions"]
		safe_prompt = code_prompt.replace("{", "{{").replace("}", "}}")
		for v in _template_vars:
			safe_prompt = safe_prompt.replace("{{" + v + "}}", "{" + v + "}")

		code_prompt_template = PromptTemplate(
			input_variables=_template_vars,
			template=safe_prompt
		)

		code_llm = ChatOpenAI(model=self.programmer_llm, temperature=0.01)
		code_chain = code_prompt_template | code_llm 

		# Generate code
		code_response = code_chain.invoke({
			"insight": insight,
			"instructions":instructions,
			"available_ids": ", ".join(df_ids),
			"input_df_summary": input_df_summary,
			"store_df": store_df
		})

		print("Generated Code:\n", code_response.content)
		
		# Return a plain dict (no Command)
		return {
			# "messages": [ToolMessage(
			# 	content="Code generated and stored in state. Ready for execution",
			# 	tool_call_id=runtime.tool_call_id
			# )],
			"messages": "Code generated and stored in state. Ready for execution",
			"generated_code": code_response.content,
			"coding_error": ""
		}

	def code_executor_fn(
		self,
		gen_code,
		runtime: ToolRuntime,
		df_ids: List[str],
		timeout: Optional[int] = EXECUTOR_TIMEOUT
	) -> Dict[str, Any]:
		"""Execute generated code on specified DataFrames. Returns a dict (not a Command)."""
		
		state = runtime.state
		# gen_code = state.get("generated_code")

		if not gen_code:
			print("No generated code found in state.")
			return {"messages": "Missing gen_code in input arguements" }# [ToolMessage(content="No code available. Generate code first.", tool_call_id=runtime.tool_call_id)]}
		
		try:
			if not df_ids:
				return {"messages": "Specify df_id or df_ids"}#[ToolMessage(content="Specify df_id or df_ids", tool_call_id=runtime.tool_call_id)]}
			
			# Collect DataFrames
			dfs = {}
			missing = []
			for df_id in df_ids:
				try:
					dfs[df_id] = self.df_store[df_id]
				except KeyError:
					missing.append(df_id)
			
			if missing:
				print(f"❌ DataFrames not found: {missing}")
				return {"messages": f"df_ids not found: {missing}"}#[ToolMessage(content=f"df_ids not found: {missing}", tool_call_id=runtime.tool_call_id)]}
			
			# Execute code
			res = run_generated_code_in_subprocess(gen_code, dfs, timeout=timeout)

			if res.get("status") == "error":
				error_msg = res.get("error")
				print("Code Execution Traceback/Error:\n", res.get("traceback",error_msg))
				return {
					"messages": f"Code execution failed with message:- {error_msg}",#[ToolMessage(content=f"Code execution failed with message:- {error_msg}", tool_call_id=runtime.tool_call_id)],
					"coding_error": error_msg,
					"coding_error_traceback": str(res.get("traceback", ""))
				}
			
			elif res.get("status") == "ok_mixed_generated":
				result_dict = res.get("result")
				response = result_dict.get("response","")
				dataframes = result_dict.get("dataframes")
				df_info_dict = {}
				message = response
				for new_df_id, values in dataframes.items():

					description = values.get("Description", "Generated dataframe")
					summary = df_summary(values['DataFrame'])
					# Store DataFrame
					self.df_store[new_df_id] = values['DataFrame']
					df_info_dict[new_df_id] = {"Summary":summary, "Description":description}
					print(f"✅ Created dataframe with id: {new_df_id} with Summary: {summary} and Description: {description}")
					message += f"\nCreated dataframe with id: {new_df_id} with Summary: {summary}"


				return {
					"messages": message,
					"dataframe_info": df_info_dict,
					"coding_error": "",
					"coding_error_traceback": ""
				}
			
			else:
				result_text = str(res.get("result", ""))
				print("Code Execution Result:\n", result_text[:500])
				return {"messages": f"Code execution result: {result_text}", 
           				"coding_error": "", "coding_error_traceback": ""}
		
		except Exception as e:
			tb = traceback.format_exc()
			print(f"Exception during code execution: {str(e)}")
			return {"messages": f"Exception during code execution: {str(e)}", "coding_error": str(e), "coding_error_traceback": tb}

	@staticmethod
	def _format_tools(tools: List[str]) -> str:
		"""Format tools list for prompt"""
		return "\n".join([f"- {tool}" for tool in tools])

	def _build_graph(self):
		def sequential_tool_node(state: WorkerState):
			tool_messages = []
			for tool_call in state["messages"][-1].tool_calls:  # Sequential loop
				tool = tools[tool_call["name"]]
				result = tool.invoke(tool_call["args"])
				tool_messages.append(ToolMessage(
					content=str(result),
					tool_call_id=tool_call["id"],
					name=tool_call["name"]
				))
			return {"messages": tool_messages}

		def data_analyst_node(state: WorkerState):
			llm = ChatOpenAI(model=self.executor_llm, temperature=0)
			
			print("\n" + "="*80)
			print("🔍 DATA ANALYST NODE INVOKED")
			if state['iteration_count'] %  (MAX_ANALYST_ITERATIONS // 2) == 0:	
				print(f"Current Task: {state['current_task']}")
			print(f"Iteration: {state['iteration_count']}/{MAX_ANALYST_ITERATIONS}")
			print("="*80)
			
			analyst_prompt = f"""You are a data analysis agent who is supposed to execute tasks given by upstream supervisor. Your output must be useful enough for further reasoning. Use the available tools and 
								dataframes stored in df_store.
			
								GENERAL INSTRUCTIONS:-
								1. Use code_generator_tool_fn only when it is impossible to get anything useful from the output of other tools.
								2. STRICTLY follow USAGE INSTRUCTIONS of all tools if available.
								3. Make a rough plan before starting execution. 
        						4. Your primary function - Supervisor delegated Current task. However, think long term based on the Supervisor's objective of :- '{self.objective}'. 
            						Eg.:- Store intermediate dataframes in the DataFrame store which will be used repeatdly.

								CRITICAL RULES:
								1. Every numeric or factual answer must come from data or Observation.
								2. Summarize your actions and results but DO NOT suggest next steps in final answer.
								3. NEVER attempt to solve for the Suervisor's objective. That is just additional context for you.
								4. Always provide summaries or head(5) and not entire Dataframes unless exlicitly mentioned.  

								Include reasoning/thought process behind particular action in your response.
								
								Current Task: {state['current_task']}

			"""

			# print(f" Last 3 messages in history:")
			# for i, msg in enumerate(state["messages"][-3:]):
			# 	print(f" Message {i}: {type(msg).__name__} - {msg.content[:100]}")

			sys_msg = SystemMessage(content=(analyst_prompt))
			messages = [sys_msg] + state["messages"]
			llm_with_tools = llm.bind_tools(self.data_analyst_tools)
			
			# The LLM returns an AIMessage. 
			# If it calls a tool, this message contains `tool_calls`.
			# If it's reasoning, it contains `content`.
			
			if state.get("iteration_count") < MAX_ANALYST_ITERATIONS:
				response = llm_with_tools.invoke(messages)
				print(f"💭 Analyst Response Content:\n{response.content}")
				if getattr(response, "tool_calls", None):
					print(f"🛠️  Tool Calls Detected: {len(response.tool_calls)}")
					for tc in response.tool_calls:
						print(f"   - Tool: {tc['name']}")
						for arg, value in tc.get('args', {}).items():
							print(f"      {arg}: {value}")
					messages_out = [AIMessage(
										content=(
											f"Reasoning: {response.content},\n"
											+ "Tool Calls: "
											+ json.dumps(
												[{"Name": tc["name"], "args": tc.get("args", {})} for tc in response.tool_calls],
												indent=2,
											)
										),
										tool_calls=response.tool_calls,
									)
								]
				else:
					messages_out = [AIMessage(content=f"Final Answer:\n{response.content}")]
			else:
				messages = messages + [SystemMessage(content="""Maximum analyst iterations reached. Summarize from whatever results 
										 you have and specify the pending tasks that couldn't be completed. No more tool calls.""")]
				response = llm_with_tools.invoke(messages)
				messages_out = [response]
				print(f"Max Analyst iterations. Analyst Final Response Content:\n{response.content}")
				# response = AIMessage(content="Maximum analyst iterations reached. Ending analysis. Give smaller/simpler tasks")
				
			return {"messages": messages_out, "iteration_count": state["iteration_count"] + 1}

		# Define the Analyst Subgraph
		analyst_builder = StateGraph(WorkerState)
		analyst_builder.add_node("reasoner", data_analyst_node)
		analyst_builder.add_node("tools", ToolNode(self.data_analyst_tools))
		# analyst_builder.add_node("tools", sequential_tool_node)

		analyst_builder.add_edge(START, "reasoner")

		def analyst_router(state):
			# If the last message has tool calls, go to tools
			last_msg = state["messages"][-1]
			if last_msg.tool_calls and state['iteration_count'] <= MAX_ANALYST_ITERATIONS:
				return "tools"
			# Otherwise end
			return END

		analyst_builder.add_conditional_edges("reasoner", analyst_router, {"tools": "tools", END: END})
		analyst_builder.add_edge("tools", "reasoner") # Loop back after tool execution
		data_analyst_graph = analyst_builder.compile()

		# --- 4. SUPERVISOR NODE (With Persistence Fix) ---

		def supervisor_node(state: GlobalState):
			
			print("\n" + "="*80)
			print("👔 SUPERVISOR NODE INVOKED")
			print(f"Iteration: {state['iteration_count']}/{MAX_ITERATIONS}")
			print("="*80)
			
			llm = ChatOpenAI(model=self.supervisor_llm, temperature=0.025)
			
			# --- BUILDING THE SEQUENTIAL MISSION LOG ---
			mission_log = []
			
			# We iterate through history to find previous Supervisor decisions
			# and combine them with the reports received.
			report_idx = 0
			for msg in state["messages"]:
				if isinstance(msg, AIMessage) and "REASONING:" in msg.content:
					mission_log.append(msg.content)
					# If a report was generated after this decision, pair it
					if report_idx < len(state.get("DataAnalyst_reports", [])):
						mission_log.append(f"RESULT: {state['DataAnalyst_reports'][report_idx]}")
						report_idx += 1

			conversation_history = "\n\n".join(mission_log)
			available_dfs = json.dumps(state.get("dataframe_info", {}), indent=2)
			tools_summary = self._format_tools([_tool_name(fn) for fn in self.data_analyst_tools])
			
			DOMAIN_KNOWLEDGE_PROMPT = f""" 
									1. The hallmark of a good forecast is it having lower absolute error over the full scope.
									2. The absolute error calculation for goodness of forecast assesment will be done at the Intersection Level :- {self.intersection_level}.
									3. Cycle of forecast refers to the date of forecast generation. Only gives temoral information and nothing about the intersections.
									"""

			DS_ML_KNOWLEDGE_PROMPT = """
									1. Possbility or ease of improving forecasts decreases with increasing accuracy.
									2. Maximum achievable accuracy decreases with hierarchical as well as temporal level.
									"""
			prompt = f"""You are a supervisor (reasoner). Do NOT generate code. Your job is to solely answer the user query, 
							Your solution must be entirely data driven. You have a data analysis agent who can give you necessary insights for your final response.
							The outcome of the alloted task must be the desired information (in summarized report format and not raw data structures) required to make subsequent decisions. 
       						Distinctly mention the requirement and don't specify the intermediate steps.
							
							AVAILABLE DATAFRAMES:
							{available_dfs}
							
							AVAILABLE TOOLS (Analyst will use these):
							{tools_summary}
							
							Supply-chain knowledge: 
							{DOMAIN_KNOWLEDGE_PROMPT}

							Data Science and ML knowledge:
							{DS_ML_KNOWLEDGE_PROMPT}

							REASONING INSTRUCTIONS:-
							1. Always utilize the Supply-chain and DS-ML knowledge provided above.
							2. All strategic (analysis approach) decisions must be taken by you as the planner. Only leave operational ones for the Analyst agent.
							3. Take baby steps! Understand the data, give small tasks to analyst, get results, reason on them and repeat until you can answer the query.
							4. The output from each step must give you concise and relevant information to make next decision.
							5. Steps to follow:-
								a. Decide what you wish to accomplish in this step ( information / insight / analysis ).
								b. Make a high level plan of how it will be achieved, what would be the approach.
								c. Ask the Analyst to perform only and only the necessary steps. Avoid unnecessary work.

							CRITICAL RULES:-
							1. Assign a single task or activity at a time and wait for Analyst's report before proceeding to the next.
							2. Keep the tasks small and focused. Avoid complex or multi-part tasks.
							3. Avoid asking for entire Dataframes to be printed as string unless absolutely required.

							OBJECTIVE:- {self.objective}

							{conversation_history}"""

			system_msg = SystemMessage(content=(prompt))
			
			# Force structured output
			structured_llm = llm.with_structured_output(SupervisorDecision)
			decision = structured_llm.invoke([system_msg])
			
			print(f"\n✅ Supervisor Decision:")
			print(f"   - Next Step: {decision.next_step}")
			print(f"   - Reasoning: {decision.reasoning[:2000]}...")
			
			if decision.next_step == "FINISH":
				print(f"   - Final Answer: {decision.answer[:1000]}...")
				ai_msg = AIMessage(content=f"REASONING: {decision.reasoning}")
				return {"next_step": "FINISH", "query_response": decision.answer,"messages": [ai_msg],"iteration_count": state["iteration_count"] + 1}
			# elif isinstance(decision, AnalysisRequest):
			else:
				print(f"   - Tasks Delegated: {decision.task}")
				if type(decision.task) is str:
					tasks = [decision.task]
				else:
					tasks = decision.task
				ai_msg = AIMessage(content=f"REASONING: {decision.reasoning}\nNext Step: {decision.next_step}\nTASKS DELEGATED: {', '.join(tasks)}")
				return {"next_step": "DataAnalyst", "messages": [ai_msg], "task_lists": [tasks], "iteration_count": state["iteration_count"] + 1}

		# --- 5. BRIDGE FUNCTION ---
		def call_data_analyst(state: GlobalState):
			# Map Global -> Worker
			last_task_list = state["task_lists"][-1] if state.get("task_lists") else []
			print(f"📋 Tasks to Execute: {len(last_task_list)} tasks")
			
			task_responses = []
			df_info = state.get("dataframe_info", {})
			
			for i,task in enumerate(last_task_list, 1):
				# Map GlobalState -> WorkerState
				worker_inputs: WorkerState = {
					"messages": [],
					"current_task": task,
					"dataframe_info": self.make_dataframe_info() if len(df_info) == 0 else df_info,
					"generated_code": "",
					"coding_error": "",
					"coding_error_traceback": "",
					"iteration_count": 0
				}
	
				# Execute Subgraph
				final_worker_state = data_analyst_graph.invoke(worker_inputs)
				df_info = final_worker_state.get("dataframe_info", {})
				# Extract Final Output
				final_response = final_worker_state["messages"][-1].content
				task_responses.append(f"Task {i} Result: {final_response}")
				print(f"   ✓ Task {i} Completed")
				
			print(f"\n✅ All {len(last_task_list)} tasks completed")
			return {
				"DataAnalyst_reports": [task_responses],
				"dataframe_info": final_worker_state.get("dataframe_info", {})
			}

		global_builder = StateGraph(GlobalState)
		global_builder.add_node("supervisor", supervisor_node)
		global_builder.add_node("DataAnalyst", call_data_analyst)
		global_builder.add_edge(START, "supervisor")

		def global_router(state):
			next_step = state.get("next_step", "")
			# FINISH or unknown routes to END
			if next_step == "DataAnalyst" and state.get("iteration_count", 0) < MAX_ITERATIONS:
				return "DataAnalyst"
			return END

		global_builder.add_conditional_edges("supervisor", global_router, {"DataAnalyst": "DataAnalyst", END: END})
		global_builder.add_edge("DataAnalyst", "supervisor") # Always report back

		return global_builder.compile()
	
	def analyze(self, dataframes: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
		"""Run analysis on user query
		
		Args:
			query: User query string
			dataframes: Optional dict of {df_id: {'DataFrame': df, 'Description': str}} to override self.input_dataframes
		"""
		if dataframes is not None:
			self.input_dataframes = dataframes
		if not self.input_dataframes:
			raise ValueError("No DataFrames provided")
		
		# Initialize dataframe_info from input_dataframes
		dataframe_info = self.make_dataframe_info()
		
		available_tools = [_tool_name(fn) for fn in self.data_analyst_tools]
		
		initial_state: GlobalState = {
			"messages": [],
			"available_tools": available_tools,
			"requirements": [],
			"DataAnalyst_reports": [],
			"next_step": "",
			"dataframe_info": dataframe_info,
			"iteration_count": 0,
			"query_response": ""
		}
		
		final_state = self.graph.invoke(initial_state)
		return {
			"query_response": final_state.get("query_response"),
			"messages": final_state.get("messages"),
			"DataAnalyst_reports": final_state.get("DataAnalyst_reports"),
			"dataframe_info": final_state.get("dataframe_info"),
			"iteration_count": final_state.get("iteration_count")
		}

if __name__ == "__main__":
	import pandas as pd
	try:
		mp.set_start_method('spawn')
	except RuntimeError:
		pass
	results = pd.read_csv(r"C:\Users\Preet Lodaya\MLOps_Bot\Results\Stat_forecast_20120426_select_articles.csv", parse_dates=['date'])
	results['month'] = results['date'].dt.to_period('M')
	results['quarter'] = results['date'].dt.to_period('Q')
	results['year'] = results['date'].dt.to_period('Y')
	results.drop('intersection', axis=1, inplace=True)

	df_dict = {
		"Forecasts_Actuals_merged": {
			"DataFrame": results,
			"Description": """Dataframe containing Actuals and forecasts (generated at {','.join(intersection_level)}) merged. Test/forecast start date :- {forecast_date}. 
				Predictions during training period also provided."""
		}
	}

	maas = MultiAgentAnalysisSystem(dataframes=df_dict)

	from langchain_community.callbacks import get_openai_callback
	with get_openai_callback() as cb:
		result = maas.analyze()
		print(f"Total Tokens: {cb.total_tokens}")
		print(f"Prompt Tokens: {cb.prompt_tokens}")
		print(f"Completion Tokens: {cb.completion_tokens}")
		print(f"Total Cost (USD): ${cb.total_cost}")

## TODO:-
# 
# 1. pass data paths instead of physical dataframes in code_execution
# 2. Paralellize code_gen_and_run (multiple calls in single response not supported)
# 3. allow both storing of dataframes and producing string outputs in code_run_tool_fn

# COST CONTROL MECHANISMS
# 1. format of output of analyst for effective communication with supervisor
# 1.a. add dataframe reading capability to SUpervisor directly
# 2. Add standard nomenclatures like AE:- Absolute Error to calculate_accuracy function to increase its utilization
# and numerical (thresholds, metrics) 

# ASSUMPTION:-
# 1. Implicit assumption that if an error occurs in code_gen_and_run, agent will call the tool aggain with same or very similar goal
# 	- Will be solved with paalellization of the tool probably 

# Produce a concise list of Tasks the analyst should execute. 
# The supervisor can only read your final responses. Neither can it see your intermediate steps nor access the dataframes. 