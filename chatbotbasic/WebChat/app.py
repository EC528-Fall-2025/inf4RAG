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
# Service endpoints (match your 3 terminals)
# =========================
RAG_BASE = os.getenv("RAG_BASE", "http://127.0.0.1:8001")
RAG_UPLOAD_FIELD = os.getenv("RAG_UPLOAD_FIELD", "file")  # or "files"

MODEL_API_BASE = os.getenv("MODEL_API_BASE", "http://127.0.0.1:8000/v1")
MODEL_API_KEY = os.getenv("MODEL_API_KEY", "ec528")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen-4b-instruct")

# Agent settings
AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "5"))
BROWSE_MAX_CHARS = int(os.getenv("BROWSE_MAX_CHARS", "1500"))
RAG_MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "6000"))

# =========================
# Logging
# =========================
LOG_FILE = os.getenv("WEBCHAT_LOG", "/home/ubuntu/inf4RAG/logs/webchat.log")
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
# Low-level LLM call (OpenAI-compatible chat via vLLM)
# =========================================================
def _llm_chat(messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 512) -> str:
    headers = {"Authorization": f"Bearer {MODEL_API_KEY}"}
    payload = {"model": MODEL_NAME, "messages": messages, "temperature": float(temperature), "max_tokens": int(max_tokens)}
    r = requests.post(f"{MODEL_API_BASE}/chat/completions", headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(data, ensure_ascii=False)


# =========================================================
# Tools: search, browse, calc, rag
# =========================================================
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def tool_search(query: str, k: int = 5) -> List[Dict[str, str]]:
    """
    Web search using DuckDuckGo HTML endpoint (no key required).
    Returns a list of {title, url, snippet}.
    """
    try:
        url = "https://duckduckgo.com/html/"
        params = {"q": query}
        resp = requests.get(url, params=params, headers={"User-Agent": UA}, timeout=30)
        resp.raise_for_status()
        html_text = resp.text
        if BeautifulSoup:
            soup = BeautifulSoup(html_text, "html.parser")
            results = []
            for a in soup.select("a.result__a")[:k]:
                title = a.get_text(" ", strip=True)
                href = a.get("href", "")
                # DuckDuckGo wraps redirects; keep as-is (still browsable)
                snippet_tag = a.find_parent("div", class_="result")
                snippet = ""
                if snippet_tag:
                    s = snippet_tag.select_one(".result__snippet")
                    if s:
                        snippet = s.get_text(" ", strip=True)
                results.append({"title": title, "url": href, "snippet": snippet})
            if results:
                return results
        # Fallback: naive regex if bs4 not available
        items = re.findall(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html_text, re.I)
        out = []
        for href, title_html in items[:k]:
            out.append({"title": html.unescape(re.sub("<.*?>", "", title_html)), "url": href, "snippet": ""})
        return out
    except Exception:
        logging.error("tool_search failed\n%s", _traceback_text())
        return []


def _extract_text_from_html(html_text: str, max_chars: int) -> str:
    if BeautifulSoup:
        soup = BeautifulSoup(html_text, "html.parser")
        # Remove script/style
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        text = soup.get_text("\n", strip=True)
    else:
        # Minimal fallback
        text = re.sub("<script.*?</script>", " ", html_text, flags=re.S | re.I)
        text = re.sub("<style.*?</style>", " ", text, flags=re.S | re.I)
        text = re.sub("<.*?>", " ", text)
        text = html.unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:max_chars]


def tool_browse(url: str, max_chars: int = BROWSE_MAX_CHARS) -> str:
    """
    Fetch a web page and return human-readable text (truncated).
    """
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
        r.raise_for_status()
        return _extract_text_from_html(r.text, max_chars)
    except Exception:
        logging.error("tool_browse failed for url=%s\n%s", url, _traceback_text())
        return ""


# Safe calculator using AST whitelist
import ast


class _CalcEval(ast.NodeVisitor):
    ALLOWED_NODES = {
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Load, ast.Pow,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.USub, ast.UAdd,
        ast.Call, ast.Name, ast.Constant
    }
    ALLOWED_FUNCS = {
        "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp,
        "sin": math.sin, "cos": math.cos, "tan": math.tan, "asin": math.asin, "acos": math.acos, "atan": math.atan,
        "ceil": math.ceil, "floor": math.floor, "fabs": math.fabs
    }
    ALLOWED_CONSTS = {"pi": math.pi, "e": math.e}

    def __init__(self):
        self._value = None

    def visit(self, node):
        if type(node) not in self.ALLOWED_NODES:
            raise ValueError(f"Disallowed expression: {type(node).__name__}")
        return super().visit(node)

    def visit_Expression(self, node):
        self._value = self.visit(node.body)

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only int/float constants are allowed")

    def visit_Num(self, node):  # for Python <3.8 compatibility
        return node.n

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            return left ** right
        raise ValueError("Unsupported binary operator")

    def visit_UnaryOp(self, node):
        val = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +val
        if isinstance(node.op, ast.USub):
            return -val
        raise ValueError("Unsupported unary operator")

    def visit_Name(self, node):
        if node.id in self.ALLOWED_CONSTS:
            return self.ALLOWED_CONSTS[node.id]
        raise ValueError(f"Unknown name: {node.id}")

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are allowed")
        name = node.func.id
        if name not in self.ALLOWED_FUNCS:
            raise ValueError(f"Function not allowed: {name}")
        args = [self.visit(a) for a in node.args]
        return self.ALLOWED_FUNCS[name](*args)


def tool_calc(expr: str) -> str:
    try:
        tree = ast.parse(expr, mode="eval")
        ev = _CalcEval()
        result = ev.visit(tree)
        if isinstance(result, (int, float)):
            return str(result)
        return str(result)
    except Exception as e:
        return f"CalcError: {e}"


def tool_rag(query: str, top_k: int = 4) -> Dict[str, Any]:
    """
    Call /rag/query and return the JSON response.
    """
    try:
        payload = {"query": query, "top_k": top_k}
        r = requests.post(f"{RAG_BASE}/rag/query", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception:
        logging.error("tool_rag failed\n%s", _traceback_text())
        return {"error": "rag_call_failed"}


def _rag_context(resp: Dict[str, Any], max_chars: int = RAG_MAX_CONTEXT_CHARS) -> str:
    """
    Extract readable context from rag response.
    """
    if not isinstance(resp, dict):
        return ""
    pieces: List[str] = []
    docs = resp.get("retrieved_documents") or resp.get("documents") or []
    if isinstance(docs, list):
        for d in docs:
            if isinstance(d, dict):
                t = d.get("text") or d.get("content") or ""
                if t:
                    pieces.append(str(t))
    if not pieces and isinstance(resp.get("context"), str):
        pieces.append(resp["context"])
    ctx = "\n\n".join(pieces).strip()
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
    "  - JSON schema: {\"tool\": \"search|browse|calc|rag|finish\", \"input\": \"...\"}\n"
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
    candidates = re.findall(r'(\{.*\})', text.strip(), flags=re.S)
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
        obs_history = []
        for i, st in enumerate(steps, 1):
            obs_history.append(f"Step {i} -> Tool: {st['tool']} | Input: {st['input']} | Observation: {st['observation']}")
        history_text = "\n".join(obs_history)

        messages = [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": f"Question: {question}\n\nPrevious steps:\n{history_text if history_text else '(none)'}\n\nReturn only a JSON with your next action."},
        ]
        model_out = _llm_chat(messages, temperature=0.0, max_tokens=256)
        action = _extract_last_json(model_out)

        if not action:
            # Fallback: attempt to finish with a generic answer
            logging.warning("Agent could not parse JSON action. Model output: %s", model_out)
            return "I could not determine the next action. Please rephrase your question."

        tool = str(action.get("tool", "")).lower().strip()
        tool_input = str(action.get("input", "")).strip()

        # Execute the selected tool
        observation = ""
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
                rag_resp = tool_rag(tool_input, top_k=4)
                if "error" in rag_resp:
                    observation = f"RAG error: {rag_resp.get('error')}"
                else:
                    ctx = _rag_context(rag_resp, max_chars=RAG_MAX_CONTEXT_CHARS)
                    # Keep a short summary for loop; full context will be used implicitly across steps
                    observation = ctx[:500] if ctx else "No context returned by RAG."
        else:
            observation = f"Unknown tool: {tool}"

        steps.append({"tool": tool, "input": tool_input, "observation": observation})

        # If the observation is empty repeatedly, break to avoid infinite loop
        if step == max_steps:
            break

    # Finalization: ask the model to produce a final answer using all Observations
    evidence = []
    for st in steps:
        evidence.append(f"- Tool={st['tool']}, Input={st['input']}, Observation={st['observation'][:300]}")
    evidence_text = "\n".join(evidence)
    messages = [
        {"role": "system", "content": FINAL_SYSTEM},
        {"role": "user", "content": f"Question: {question}\n\nObservations:\n{evidence_text}\n\nWrite the final answer in English."},
    ]
    return _llm_chat(messages, temperature=0.2, max_tokens=600)


# =========================================================
# UI helpers: RAG upload and chat handlers
# =========================================================
def toggle_rag_panel(checked: bool):
    return gr.update(visible=bool(checked))


def upload_zip_to_rag(file_obj, dataset_name: Optional[str]):
    """
    Forward .zip to /rag/upload, tolerant to gr.File returning a str or object with .name.
    """
    try:
        if not file_obj:
            return gr.update(), "Please choose a .zip file."
        dataset_name = dataset_name or "default-dataset"
        if isinstance(file_obj, (str, os.PathLike)):
            path = str(file_obj)
        else:
            path = getattr(file_obj, "name", None) or str(file_obj)
        if not os.path.exists(path):
            logging.error("Upload failed: temp file path not exists: %s", path)
            return gr.update(), f"Upload failed: temp file not found: {path}"

        def _post(field_name: str):
            with open(path, "rb") as f:
                files = {field_name: (os.path.basename(path), f, "application/zip")}
                data = {"dataset": dataset_name}
                return requests.post(f"{RAG_BASE}/rag/upload", files=files, data=data, timeout=180)

        r = _post(RAG_UPLOAD_FIELD)
        if not r.ok:
            alt = "files" if RAG_UPLOAD_FIELD == "file" else "file"
            r2 = _post(alt)
            if not r2.ok:
                logging.error("Upload failed: %s %s; alt: %s %s", r.status_code, r.text[:300], r2.status_code, r2.text[:300])
                return gr.update(), f"Upload failed: {r2.status_code} {r2.text[:300]}"
            try:
                j = r2.json()
                msg = j.get("message") or j.get("detail") or r2.text[:300]
            except Exception:
                msg = r2.text[:300]
            return gr.update(value=None), f"Upload successful: {msg}"

        try:
            j = r.json()
            msg = j.get("message") or j.get("detail") or r.text[:300]
        except Exception:
            msg = r.text[:300]
        return gr.update(value=None), f"Upload successful: {msg}"

    except Exception:
        logging.error("Upload error\n%s", _traceback_text())
        return gr.update(), "Upload failed. See server logs for traceback."


def chat_fn(history: list, user_msg: str, rag_on: bool):
    """
    Chat entry. Always uses the agentic workflow.
    """
    if not user_msg:
        return history, ""
    try:
        answer = agentic_answer(user_msg, rag_enabled=rag_on, max_steps=AGENT_MAX_STEPS)
        history = history + [(user_msg, answer)]
        return history, ""
    except Exception as e:
        logging.error("chat_fn failed: %s\n%s", e, _traceback_text())
        history = history + [(user_msg, f"Error: {e.__class__.__name__}: {e}")]
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
        upload = gr.File(label="Upload RAG Documents (.zip)", file_types=[".zip"], file_count="single")
        dataset = gr.Textbox(label="Dataset name", value="default-dataset")
        with gr.Row():
            upload_btn = gr.Button("Upload")
            status = gr.Markdown()

    gr.Markdown("#### Chat")
    chatbot = gr.Chatbot(label="Chatbot", height=420, type="tuples")
    with gr.Row():
        msg = gr.Textbox(placeholder="Type your question and press Enter", scale=4)
        send = gr.Button("Send", variant="primary")
        clear = gr.Button("Clear")

    # Bind events
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
