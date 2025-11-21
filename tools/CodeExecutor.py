
import json
import uuid
import traceback
import ast
import multiprocessing as mp
from multiprocessing import Queue, Process
from typing import Dict, Any
import time
import logging
from langchain.tools import tool
import pandas as pd, numpy as np
import builtins

# Configure logging in the main part of your script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
)

DANGEROUS_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "os", "sys", "subprocess",
    "socket", "ftplib", "shutil", "pathlib", "requests", "http", "urllib"
}
DANGEROUS_ATTR_PREFIX = "__"

def ast_safety_check(code: str):
    """
    Basic AST-based checks:
      - reject Import and ImportFrom nodes
      - reject ClassDef, AsyncFunctionDef
      - reject usage of dangerous names (open, exec, eval, __import__, os, sys, subprocess...)
      - reject attr names starting with __ (dunder attribute access)
    Throws ValueError if checks fail.
    """
    tree = ast.parse(code)
    for node in ast.walk(tree):
        # if isinstance(node, (ast.Import, ast.ImportFrom)):
        #     raise ValueError("Imports are not allowed in generated code.")
        # if isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef)):
        #     raise ValueError("Defining classes or async functions is not allowed.")
        if isinstance(node, ast.Call):
            # check for direct calls to dangerous names
            func = node.func
            if isinstance(func, ast.Name) and func.id in DANGEROUS_NAMES:
                raise ValueError(f"Use of dangerous function/name '{func.id}' is not allowed.")
            if isinstance(func, ast.Attribute) and getattr(func.attr, "startswith", lambda *_: False)(DANGEROUS_ATTR_PREFIX):
                raise ValueError("Calling dunder attributes is not allowed.")
        if isinstance(node, ast.Attribute):
            if isinstance(node.attr, str) and node.attr.startswith(DANGEROUS_ATTR_PREFIX):
                raise ValueError("Dunder attribute access is not allowed.")
        if isinstance(node, ast.Name) and node.id in DANGEROUS_NAMES:
            raise ValueError(f"Reference to dangerous name '{node.id}' is not allowed.")

# -------------------------
# Safe execution in separate process
# -------------------------
def _worker_exec(code_str: str, df: pd.DataFrame, queue: Queue):
    """
    Worker: executed in a separate process.
    - Unpickle df (already pickled by Process args)
    - Exec code_str in restricted environment (limited builtins)
    - If transform(df) exists -> call it
      else if 'result' variable exists -> use it directly
    - Send result or detailed error info to queue
    """
    try:
        logging.info("Starting execution...")
        print("[Worker] Code to execute:\n", code_str, flush=True)
        
        
        def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            # allow only a small set of safe modules (expand carefully if needed)
            _real_import = builtins.__import__
            
            allowed_root = {
                "math", "statistics", "datetime", "json", "re",
                "itertools", "functools", "operator", "collections",
                "pandas", "numpy"
            }
            root = name.split(".")[0]
            
            if root in allowed_root:
                # if module already loaded use that, else delegate to real import
                return _real_import(name, globals, locals, fromlist, level)
            raise ImportError(f"Import of module '{name}' is restricted.")

        allowed_builtins_names = [
            "abs", "all", "any", "bool", "chr", "complex", "dict", "divmod",
            "enumerate", "float", "int", "len", "list", "map", "max", "min",
            "next", "pow", "range", "repr", "round", "sorted", "sum", "zip",
            "print"
        ]
        safe_builtins = {name: getattr(builtins, name)
                            for name in allowed_builtins_names if hasattr(builtins, name)}
        
        # Create globals with pandas and numpy available
        restricted_globals = {"pd": pd, "np": np, "__builtins__": safe_builtins}
        # inject safe __import__ so import statements in generated code don't fail
        safe_builtins["__import__"] = _safe_import
        # Local namespace for exec
        local_ns = {"df": df}
        logging.info("[Worker] DataFrame injected into local_ns.")

        # Execute the user-provided code
        logging.info("[Worker] Executing code...")
        exec(code_str, restricted_globals, local_ns)
        logging.info("[Worker] Execution completed successfully.")

        # --- Flexible output handling ---
        if "transform" in local_ns:
            print("[Worker] Found transform(df) function, calling it...")
            result = local_ns["transform"](df)
        elif "result" in local_ns:
            print("[Worker] Found 'result' variable, using it directly.")
            result = local_ns["result"]
        else:
            msg = "Neither transform(df) function nor result variable found after execution."
            print("[Worker] ERROR:", msg)
            queue.put({"status": "error", "error": msg})
            return

        # --- Validate result type ---
        if isinstance(result, dict):
            if isinstance(result.get("DataFrame"), pd.DataFrame):
                print("[Worker] Result is a DataFrame, sending back.")
                queue.put({"status": "ok_df_generated", "result": result})
            else:
                msg = "Result dict must contain a DataFrame under key 'DataFrame'."
                print("[Worker] ERROR:", msg)
                queue.put({"status": "error", "result": msg})
        elif isinstance(result, str):
            print("[Worker] Result is string.")
            queue.put({"status": "ok_str_generated", "result": result})
            
        else:
            print(f"[Worker] Result is of type {type(result)}, converting to string.")
            queue.put({"status": "ok_random_generated", "result": str(result)})

        print("[Worker] Successfully pushed result to queue.")

    except Exception as e:
        logging.info('here')
        tb = traceback.format_exc()
        print("[Worker] EXCEPTION OCCURRED:", e)
        logging.info(tb)
        queue.put({"status": "error", "result": str(e), 
                   "traceback": tb})


# def run_generated_code_in_subprocess(code_str: str, df: pd.DataFrame, timeout: int = 30):
#     """
#     Runs AST checks, then runs code in separate process with timeout.
#     Returns dict with status and either new_df (key 'result_df_id') or 'non_df' with serializable output.
#     """

#     # 1) static checks
#     try:
#         ast_safety_check(code_str)
#     except Exception as e:
#         return {"status": "error", "error": f"AST safety check failed: {e}"}
    
#     # 2) spawn process
#     q = mp.Queue()
#     # Note: pass df directly (it will be pickled into the subprocess)
#     p = Process(target=_worker_exec, args=(code_str, df, q), daemon=True)
    
#     p.start()
#     p.join(timeout)
#     if p.is_alive():
#         p.terminate()
#         return {"status": "error", "error": "Execution timed out."}

#     if not q.empty():
#         out = q.get()
#     else:
#         return {"status": "error", "result": "No result returned from worker (possibly killed)."}

#     if out.get("status") == "ok_df":
#         return {"status": "ok_df_generated", "result": {'DataFrame':out["df"],'df_id':out.get("df_id") }, #"result":f"New Dataframe with id {new_df_id} added to the store"
#                 }
#     elif out.get("status") == "ok_str":
#         return {"status": "ok_str_generated", "result": out.get("result")}
#     elif out.get("status") == "ok_random":
#         return {"status": "ok_random_datatype_generated", "result": out.get("result")}
#     else:
#         return {"status": "error", "result": out.get("result"), 
#                 # "traceback": out.get("traceback") ##TODO: SKIPPED FOR NOW BUT TO BE ADDED LATER
#                 }

def run_generated_code_in_subprocess(code_str: str, df: pd.DataFrame, timeout: int = 30):
    """
    Runs AST checks, then runs code in separate process with timeout.
    Returns dict with status and either new_df or serialized output.
    """
    if code_str.startswith("```"):
        # Split by lines
        lines = code_str.split("\n")
        # Remove first line (```python or ```)
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove last line (```) if present
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        # Rejoin
        code_str = "\n".join(lines).strip()
    
    try:
        ast_safety_check(code_str)
    except Exception as e:
        return {"status": "error", "error": f"AST safety check failed: {e}", "traceback": traceback.format_exc()}
    
    q = mp.Queue()
    p = Process(target=_worker_exec, args=(code_str, df, q), daemon=True)
    p.start()
    p.join(timeout)
    
    if p.is_alive():
        p.terminate()
        return {"status": "error", "error": "Execution timed out.", "traceback": None}

    if not q.empty():
        out = q.get()
    else:
        return {"status": "error", "error": "No result returned from worker.", "traceback": None}

    # Include traceback for error cases
    if out.get("status", "").startswith("ok"):
        return out
    else:
        return {
            "status": "error",
            "error": out.get("result", "Unknown error"),
            "traceback": out.get("traceback", "No traceback captured")
        }
