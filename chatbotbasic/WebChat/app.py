#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import json
import math
import time
import html
import logging
import traceback
from typing import Optional, List, Dict, Any

import requests
import gradio as gr
import yaml  # Configuration is loaded from config.yaml

# Optional dependency for web page parsing
try:
    from bs4 import BeautifulSoup  # pip install beautifulsoup4
except Exception:
    BeautifulSoup = None  # Fallback to plain text if bs4 is not available

# =========================
# Build tag to confirm which file is running
# =========================
APP_BUILD = os.getenv("APP_BUILD", "2025-11-19-agentic-rag-v1")
print(f"[WebChat] Starting build: {APP_BUILD}  __file__={__file__}", flush=True)

# =========================
# Configuration (config.yaml + environment variables)
# =========================


def _load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration.

    The file is optional: if it does not exist or cannot be parsed we
    fall back to built-in defaults and environment variables.
    """
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            logging.warning("Config file %s does not contain a mapping; ignoring.", path)
            return {}
        return data
    except FileNotFoundError:
        logging.warning("Config file %s not found; using defaults and environment variables.", path)
    except Exception as e:
        logging.warning("Failed to load config %s: %s; using defaults and environment variables.", path, e)
    return {}


# Path to config.yaml (can be overridden from the environment)
DEFAULT_CONFIG_PATH = os.getenv(
    "WEBCHAT_CONFIG",
    os.path.join(os.path.dirname(__file__), "config.yaml"),
)
CONFIG: Dict[str, Any] = _load_config(DEFAULT_CONFIG_PATH)

# =========================
# Service endpoints (match your 3 terminals)
# =========================

# --- LLM / model server ----------------------------------------------------
_openstack_ip_port = CONFIG.get("openstack_ip_port")
_default_model_api_base = (
    f"http://{_openstack_ip_port}/v1"
    if _openstack_ip_port
    else "http://127.0.0.1:8000/v1"
)

MODEL_API_BASE = os.getenv(
    "MODEL_API_BASE",
    CONFIG.get("model_api_base", _default_model_api_base),
)
MODEL_API_KEY = os.getenv("MODEL_API_KEY", CONFIG.get("api_key", "ec528"))

# --- RAG backend -----------------------------------------------------------
# RAG_BASE is the base URL *without* the /rag suffix, e.g. http://<ip>:8001
RAG_BASE = os.getenv("RAG_SERVICE_URL")  # from podman-compose.yml
RAG_UPLOAD_FIELD = os.getenv(
    "RAG_UPLOAD_FIELD",
    CONFIG.get("rag_upload_field", "file"),
)  # or "files"

# =========================
# Agent settings
# =========================
AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", str(CONFIG.get("max_actions", 5))))
BROWSE_MAX_CHARS = int(os.getenv("BROWSE_MAX_CHARS", "1500"))
RAG_MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "6000"))

# =========================
# Model auto-detection
# =========================


def autodetect_model_name(api_base: str, api_key: str, fallback: str) -> str:
    """Query the /v1/models endpoint to discover the default model name.

    This mirrors the behavior of the previous WebChat version: we call
    the OpenAI-compatible /models endpoint once at startup and keep the
    first model id. If anything goes wrong we fall back to the provided
    default name.
    """
    url = f"{api_base.rstrip('/')}/models"
    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        # OpenAI-style response: {"data": [{"id": "model-name", ...}, ...]}
        models = payload.get("data") or payload.get("models") or []
        if isinstance(models, list) and models:
            first = models[0]
            if isinstance(first, dict):
                model_id = first.get("id") or fallback
            else:
                model_id = str(first)
            logging.info("Auto-detected model from %s: %s", url, model_id)
            return model_id
    except Exception as e:
        logging.warning("Could not auto-detect model name from %s: %s", url, e)

    logging.warning("Falling back to default model name: %s", fallback)
    return fallback


DEFAULT_MODEL_NAME = CONFIG.get("default_model") or os.getenv(
    "MODEL_NAME", "qwen-4b-instruct"
)
MODEL_NAME = autodetect_model_name(
    MODEL_API_BASE, MODEL_API_KEY, fallback=DEFAULT_MODEL_NAME
)

# =========================
# Logging
# =========================
_default_log_dir = os.path.join(os.path.dirname(__file__), "logs")
LOG_FILE = os.getenv("WEBCHAT_LOG", os.path.join(_default_log_dir, "webchat.log"))
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03f %(levelname)s [%(process)d] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE, encoding="utf-8")],
)


def _excepthook(exc_type, exc, tb):
    logging.error("UNHANDLED EXCEPTION", exc_info=(exc_type, exc, tb))


sys.excepthook = _excepthook


def _traceback_text() -> str:
    et, ev, tb = sys.exc_info()
    return "".join(traceback.format_exception(et, ev, tb))


# =========================================================
# LLM helper
# =========================================================


def _llm_chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    """Call the OpenAI-compatible /chat/completions endpoint.

    This is a minimal helper that all higher-level functions use.
    """
    url = f"{MODEL_API_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {MODEL_API_KEY}"}
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    logging.info("Calling LLM %s model=%s", url, MODEL_NAME)
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        logging.error("Unexpected /chat/completions payload: %s", data)
        raise


# =========================================================
# Safe math evaluator for calc tool
# =========================================================
import ast
import operator

SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

SAFE_FUNCTIONS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
}


class SafeEval(ast.NodeVisitor):
    """Very small math-only expression evaluator.

    This is used for the "calc" tool so that the LLM cannot execute
    arbitrary Python code.
    """

    def visit(self, node):
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        return super().visit(node)

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Operator {op_type} not allowed")
        return SAFE_OPERATORS[op_type](left, right)

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Operator {op_type} not allowed")
        return SAFE_OPERATORS[op_type](operand)

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function names are allowed")
        name = node.func.id
        if name not in SAFE_FUNCTIONS:
            raise ValueError(f"Function {name} not allowed")
        args = [self.visit(arg) for arg in node.args]
        return SAFE_FUNCTIONS[name](*args)

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only int/float constants are allowed")

    def visit_Num(self, node):  # for Python <3.8 compatibility
        return node.n


def safe_eval_expr(expr: str) -> float:
    """Safely evaluate a math expression."""
    try:
        tree = ast.parse(expr, mode="eval")
        return SafeEval().visit(tree)
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")


# =========================================================
# Tools: search / browse / calc / rag
# =========================================================

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/119.0 Safari/537.36"
)


def tool_search(query: str, k: int = 5) -> List[Dict[str, str]]:
    """Very small DuckDuckGo HTML search scraper.

    Returns a list of {title, url, snippet}.
    """
    try:
        url = "https://duckduckgo.com/html"
        params = {"q": query}
        resp = requests.get(url, params=params, headers={"User-Agent": UA}, timeout=30)
        resp.raise_for_status()
        html_text = resp.text
        if BeautifulSoup:
            soup = BeautifulSoup(html_text, "html.parser")
            results: List[Dict[str, str]] = []
            for a in soup.select("a.result__a")[:k]:
                title = a.get_text(" ", strip=True)
                href = a.get("href", "")
                snippet_tag = a.find_parent("div", class_="result")
                snippet = ""
                if snippet_tag:
                    s = snippet_tag.select_one(".result__snippet")
                    if s:
                        snippet = s.get_text(" ", strip=True)
                results.append({"title": title, "url": href, "snippet": snippet})
            return results
        # Fallback: just return a small dict with the raw HTML
        return [{"title": "raw_html", "url": url, "snippet": html_text[:500]}]
    except Exception as e:
        logging.error("tool_search error: %s", e)
        return [{"title": "error", "url": "", "snippet": f"Search error: {e}"}]


def tool_browse(url: str, max_chars: int = BROWSE_MAX_CHARS) -> str:
    """Fetch a URL and return a readable text snippet."""
    try:
        resp = requests.get(url, headers={"User-Agent": UA}, timeout=30)
        resp.raise_for_status()
        text = resp.text
        if BeautifulSoup:
            soup = BeautifulSoup(text, "html.parser")
            # Remove scripts/styles
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = soup.get_text(" ", strip=True)
        text = html.unescape(text)
        return text[:max_chars]
    except Exception as e:
        logging.error("tool_browse error for %s: %s", url, e)
        return f"[browse error] {e}"


def tool_calc(expr: str) -> str:
    """Evaluate a math expression using the safe evaluator."""
    try:
        value = safe_eval_expr(expr)
        return str(value)
    except Exception as e:
        return f"[calc error] {e}"


def _rag_query(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Internal helper: call /rag/query on the RAG backend."""
    try:
        r = requests.post(f"{RAG_BASE}/rag/query", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logging.error("RAG query error: %s", e)
        return {"error": str(e)}


def _rag_context(resp: Dict[str, Any], max_chars: int = RAG_MAX_CONTEXT_CHARS) -> str:
    """Extract a compact context string from RAG query response."""
    if not resp:
        return ""
    if "error" in resp:
        return f"[RAG error] {resp['error']}"
    docs = resp.get("documents") or resp.get("chunks") or []
    ctx = ""
    for d in docs:
        text = d.get("text") or d.get("content") or ""
        if text:
            if ctx:
                ctx += "\n\n---\n\n"
            ctx += text
        if len(ctx) >= max_chars:
            break
    return ctx[:max_chars]


# =========================================================
# Agent controller (ReAct-like with strict JSON action)
# =========================================================
AGENT_SYSTEM = (
    "You are an autonomous assistant with access to tools.\n"
    "Available tools:\n"
    "  - search(query): web search, returns a list of results with title/url/snippet.\n"
    "  - browse(url): fetch a web page and return readable text.\n"
    "  - calc(expr): evaluate a math expression using + - * / // % **, parentheses, and common math functions.\n"
    "  - rag(query): retrieve context from the local RAG index.\n"
    "Rules:\n"
    "  - Use at most one tool per step.\n"
    "  - If you already have enough information, choose the 'finish' tool.\n"
    "  - Always respond with a JSON object ONLY on the last line of your message.\n"
    '  - JSON schema: {"tool": "search|browse|calc|rag|finish", "input": "..."}\n'
    "  - Do not include any other text outside of the JSON. Be concise.\n"
)

FINAL_SYSTEM = (
    "Produce a helpful final answer in plain English. Use the given Observations as evidence.\n"
    "If insufficient information, say you don't know."
)


def _extract_last_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the last JSON object from the text. Return None if parsing fails.
    """
    candidates = re.findall(r"(\{.*\})", text.strip(), flags=re.S)
    for raw in reversed(candidates):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "tool" in obj:
                return obj
        except Exception:
            continue
    return None


def _summarize_observation(obs: Any, limit: int = 500) -> str:
    """
    Convert tool observation into a short string fed back to the LLM.
    """
    if isinstance(obs, str):
        return obs[:limit]
    try:
        s = json.dumps(obs, ensure_ascii=False)
        return s[:limit]
    except Exception:
        return str(obs)[:limit]


def agentic_answer(question: str, rag_enabled: bool, max_steps: int = AGENT_MAX_STEPS) -> str:
    """
    Main agent loop: plan -> act -> observe -> repeat -> finish.
    """
    steps: List[Dict[str, Any]] = []
    for step in range(1, max_steps + 1):
        # Compose history as Observations summary
        history_text = ""
        for s in steps:
            history_text += (
                f"Step {s['step']}:\n"
                f"  Thought: {s['thought']}\n"
                f"  Action: {s['tool']}({s['tool_input']})\n"
                f"  Observation: {s['observation']}\n\n"
            )

        user_prompt = (
            f"User question: {question}\n\n"
            f"Previous steps:\n{history_text}\n"
            "Think about which tool to use next and respond with ONLY a JSON object "
            'on the last line, following the required schema.'
        )

        messages = [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        model_out = _llm_chat(messages, temperature=0.1, max_tokens=800)
        logging.info("Agent step %d raw model output: %s", step, model_out)

        action = _extract_last_json(model_out)
        if not action:
            logging.warning("Agent could not parse JSON action. Model output: %s", model_out)
            return "[agent error] Could not parse tool JSON."

        tool = (action.get("tool") or "").strip().lower()
        tool_input = (action.get("input") or "").strip()
        logging.info("Agent step %d chose tool=%s input=%s", step, tool, tool_input)

        # Execute the selected tool
        observation: Any = ""
        if tool == "finish":
            return tool_input or "Done."
        elif tool == "search":
            results = tool_search(tool_input, k=5)
            observation = _summarize_observation(results)
        elif tool == "browse":
            observation = tool_browse(tool_input, max_chars=BROWSE_MAX_CHARS)
        elif tool == "calc":
            observation = tool_calc(tool_input)
        elif tool == "rag":
            if not rag_enabled:
                observation = "RAG tool is disabled by UI."
            else:
                payload = {"query": tool_input}
                rag_resp = _rag_query(payload)
                ctx = _rag_context(rag_resp, max_chars=RAG_MAX_CONTEXT_CHARS)
                observation = ctx or "RAG returned no context."
        else:
            observation = f"[agent error] Unknown tool: {tool}"

        steps.append(
            {
                "step": step,
                "thought": model_out,
                "tool": tool,
                "tool_input": tool_input,
                "observation": observation,
            }
        )

    # After max_steps, ask the model for a final answer
    summary_obs = "\n\n".join(
        [f"Step {s['step']} OBSERVATION: {s['observation']}" for s in steps]
    )
    messages = [
        {"role": "system", "content": FINAL_SYSTEM},
        {
            "role": "user",
            "content": f"Question: {question}\n\nObservations:\n{summary_obs}",
        },
    ]
    final_answer = _llm_chat(messages, temperature=0.3, max_tokens=1024)
    return final_answer


# =========================================================
# UI helpers: RAG upload and chat handlers
# =========================================================
def toggle_rag_panel(checked: bool):
    return gr.update(visible=bool(checked))


def upload_zip_to_rag(file_obj, dataset_name: Optional[str]):
    """
    Forward .zip to /rag/upload, tolerant to gr.File returning a str
    or object with .name.
    """
    try:
        if not file_obj:
            return gr.update(), "Please choose a .zip file."
        dataset_name = dataset_name or "default-dataset"
        if isinstance(file_obj, (str, os.PathLike)):
            path = str(file_obj)
        else:
            path = getattr(file_obj, "name", None)
        if not path or not os.path.exists(path):
            return gr.update(), "Uploaded file path is invalid."

        with open(path, "rb") as f:
            files = {RAG_UPLOAD_FIELD: (os.path.basename(path), f, "application/zip")}
            data = {"dataset": dataset_name}

            def _post(field_name: str):
                # Build body according to field name expected by RAG
                files_local = {field_name: files[RAG_UPLOAD_FIELD]}
                r = requests.post(
                    f"{RAG_BASE}/rag/upload",
                    files=files_local,
                    data=data,
                    timeout=180,
                )
                r.raise_for_status()
                return r

            try:
                r = _post(RAG_UPLOAD_FIELD)
            except Exception:
                # Try alternate field name for backwards compatibility
                alt = "files" if RAG_UPLOAD_FIELD == "file" else "file"
                logging.warning(
                    "Upload with field '%s' failed, retrying with '%s'",
                    RAG_UPLOAD_FIELD,
                    alt,
                )
                r = _post(alt)
        return gr.update(value=None), f"Upload OK: {r.text}"
    except Exception as e:
        logging.error("Upload failed: %s", e)
        return gr.update(value=None), f"Upload error: {e}\n\n{_traceback_text()}"


def chat_fn(history: List[List[str]], user_msg: str, rag_on: bool):
    """
    Gradio callback: single turn chat using the agentic controller.
    """
    history = history or []
    history.append([user_msg, "..."])
    try:
        answer = agentic_answer(user_msg, rag_enabled=rag_on, max_steps=AGENT_MAX_STEPS)
    except Exception as e:
        logging.error("chat_fn error: %s", e)
        answer = f"[chat error] {e}\n\n{_traceback_text()}"
    history[-1][1] = answer
    return history, ""


def clear_chat():
    return []


# =========================================================
# Gradio UI
# =========================================================
with gr.Blocks(title="WebChat (Agentic + optional RAG)") as demo:
    gr.Markdown("### WebChat (Agentic + optional RAG)\nAgentic mode is always enabled.")

    with gr.Row():
        rag_enable = gr.Checkbox(label="Enable RAG mode", value=True)

    with gr.Column(visible=True) as rag_panel:
        upload = gr.File(
            label="Upload RAG Documents (.zip)",
            file_types=[".zip"],
            file_count="single",
        )
        dataset = gr.Textbox(label="Dataset name", value="default-dataset")
        with gr.Row():
            upload_btn = gr.Button("Upload")
            status = gr.Markdown()

    gr.Markdown("#### Chat")
    chatbot = gr.Chatbot(label="Chatbot", height=420, type="tuples")
    with gr.Row():
        msg = gr.Textbox(
            label="User message", show_label=False, placeholder="Ask something..."
        )
        send = gr.Button("Send")
    clear = gr.Button("Clear history")

    # Wiring
    rag_enable.change(toggle_rag_panel, inputs=[rag_enable], outputs=[rag_panel])
    upload_btn.click(upload_zip_to_rag, inputs=[upload, dataset], outputs=[upload, status])
    msg.submit(chat_fn, inputs=[chatbot, msg, rag_enable], outputs=[chatbot, msg])
    send.click(chat_fn, inputs=[chatbot, msg, rag_enable], outputs=[chatbot, msg])
    clear.click(clear_chat, outputs=[chatbot])

# =========================================================
# Launch
# =========================================================
demo.queue()
demo.launch(
    server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
    server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    show_api=False,
    debug=True,
)
