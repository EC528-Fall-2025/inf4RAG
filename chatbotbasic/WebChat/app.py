from __future__ import annotations


import re
import datetime
import gradio as gr
import yaml
import requests
import os


from models import OpenAIModel
from tools import Tools
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.utils.output_beautify import typewriter_print




SYSTEM_MESSAGE_TEMPLATE = "prompt.txt"


with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)


# synthesize the base_url
config["base_url"] = f"http://{config.get('openstack_ip_port', '127.0.0.1:8000')}/v1"


MODELS = {
    "vLLM": OpenAIModel(config)
}
if "qwen" in config.get("enabled_models", []):
    qwen_config = config.get("qwen", {})
    MODELS["qwen"] = Assistant(**qwen_config)



verbose = config["verbose"]
description = config["description"]
examples = config["examples"]
temperature = config["temperature"]
max_actions = config["max_actions"]


# RAG Configuration
RAG_SERVICE_URL = "http://localhost:8001"


llm_tools = Tools()


def create_system_message():
    """
    Return system message, including today's date and the available tools.
    """
    with open(SYSTEM_MESSAGE_TEMPLATE) as f:
        message = f.read()


    now = datetime.datetime.now()
    current_date = now.strftime("%B %d, %Y")


    message = message.replace("{{CURRENT_DATE}}", current_date)


    return message


def check_rag_service():
    """Check if RAG service is available"""
    try:
        response = requests.get(f"{RAG_SERVICE_URL}/rag/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def upload_file_to_rag(file, dataset_id):
    """Upload file to RAG service"""
    if file is None:
        return "No file selected"
   
    try:
        # Read the file
        with open(file.name, 'rb') as f:
            files = {'file': f}
            data = {'dataset_id': dataset_id}
           
            response = requests.post(
                f"{RAG_SERVICE_URL}/rag/upload",
                files=files,
                data=data,
                timeout=120
            )
       
        if response.status_code == 200:
            result = response.json()
            return f"✓ Upload successful! Processed {result.get('file_count', 0)} files to dataset '{dataset_id}'"
        else:
            return f"✗ Upload failed: {response.text}"
           
    except requests.exceptions.ConnectionError:
        return f"✗ Error: Cannot connect to RAG service at {RAG_SERVICE_URL}. Please ensure RAG service is running."
    except Exception as e:
        return f"✗ Error: {str(e)}"


def generate(new_user_message, history, use_rag=False):
    # Only use RAG if enabled
    if use_rag:
        try:
            response = requests.post(
                f"{RAG_SERVICE_URL}/rag/query",
                json={
                    "query": new_user_message,
                    "dataset_id": "default-dataset",
                    "top_k": 4
                },
                timeout=30
            )
           
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    # Use RAG-generated prompt
                    prompt = result["prompt"]
                    if verbose:
                        print(f"[RAG] Using retrieved context with {result['retrieved_count']} documents")
                else:
                    prompt = new_user_message
            else:
                prompt = new_user_message
        except Exception as e:
            if verbose:
                print(f"[RAG] Error: {e}")
            prompt = new_user_message
    else:
        # Normal mode - use original query
        prompt = new_user_message


    # Collect full response from stream
    full_response = ""
    iters = 0
    model = MODELS["vLLM"]


    try:
        while True:
            if verbose:
                print("="*80)
                print(f"ITERATION {iters}")
                print("="*80)
                print(prompt)


            stream = model.generate(
                system_message,
                prompt,
                history=history,
                temperature=temperature
            )


            # Collect all chunks into full_response
            for chunk in stream:
                completion = model.parse_completion(chunk)
                if completion:
                    full_response += completion
                    yield full_response  


                    tool_match = re.search(r"<tool_call>(.*?)</tool_call>", completion)
                    if tool_match:
                        tool_name = tool_match.group(1).strip()
                        if verbose:
                            print(f"🔧 (AGENTIC) Detected tool call: {tool_name}")


                         #GoogleSearch agentic tool
                        if tool_name.lower() == "googlesearch":
                            query_match = re.search(r"<query>(.*?)</query>", completion)
                            query = query_match.group(1) if query_match else prompt
                            result = llm_tools.google_search(query)


                            # Append tool result to prompt for next iteration
                            prompt += f"\n<RESULT>{result}</RESULT>"


                            # Break the inner stream loop to restart generation with tool result
                            break


           
            # Exit the while loop after processing the stream
            break
   
    except Exception as e:
        full_response += f"\n<span style='color:red'>Error: {e}</span>"
   
    # Return the final response string instead of a generator
    return full_response




def on_rag_mode_toggle(button_state):
    """Handle RAG mode toggle button click"""
    global current_rag_mode
   
    if current_rag_mode:
        # Currently in RAG mode, switch to normal mode
        current_rag_mode = False
        return (
            gr.Button(value="Enable RAG Mode", variant="secondary"),
            gr.File(visible=False),
            gr.Textbox(visible=True, value="Normal conversation mode. Click to enable RAG mode.")
        )
    else:
        # Currently in normal mode, check service and switch to RAG mode
        if not check_rag_service():
            return (
                gr.Button(value="Enable RAG Mode", variant="secondary"),
                gr.File(visible=False),
                gr.Textbox(visible=True, value="⚠️ RAG service is not available. Please start the RAG service at http://localhost:8001")
            )
        current_rag_mode = True
        return (
            gr.Button(value="Disable RAG Mode", variant="primary"),
            gr.File(visible=True),
            gr.Textbox(visible=True, value="RAG mode enabled. You can upload documents.")
        )




# Create Gradio app
system_message = create_system_message()
if verbose:
    print("="*80)
    print("SYSTEM PROMPT:")
    print("="*80)
    print(system_message)


CSS = """
h1 { text-align: center; }
h3 { text-align: center; }
.contain { display: flex; flex-direction: column; }
.gradio-container { height: 100vh !important; }
#component-0 { flex-grow: 1; overflow: auto; }
#chatbot { flex-grow: 1; overflow: auto; }
body {
    min-height: 100vh;
}
"""


# Global state for RAG mode
current_rag_mode = False


with gr.Blocks(css=CSS) as app:
    gr.Markdown(description)
   
    # RAG mode toggle
    with gr.Row():
        use_rag_button = gr.Button(
            value="Enable RAG Mode",
            variant="secondary"
        )
   
    # Upload section - initially hidden
    file_upload = gr.File(
        label="Upload RAG Documents (.zip)",
        file_count="single",
        file_types=[".zip"],
        visible=False
    )
   
    upload_status = gr.Textbox(
        label="Status",
        value="Normal conversation mode. Click 'Enable RAG Mode' to upload documents.",
        interactive=False,
        visible=False
    )
   
    # RAG mode toggle handler
    use_rag_button.click(
        fn=on_rag_mode_toggle,
        inputs=None,
        outputs=[use_rag_button, file_upload, upload_status],
        show_progress=False
    )
   
    # File upload handler
    file_upload.change(
        fn=lambda file: upload_file_to_rag(file, "default-dataset"),
        inputs=file_upload,
        outputs=upload_status
    )
   
    # Chat interface
    def chat_handler(message, history):
        global current_rag_mode
        return generate(message, history, use_rag=current_rag_mode)
   
    chatinterface = gr.ChatInterface(
        fn=chat_handler,
        examples=examples
    )
    chatinterface.chatbot.elem_id = "chatbot"


    with gr.Accordion(label="Options", open=False):
        with gr.Row():
            model_name_box = gr.Textbox(
                label="Model Name",
                placeholder="N/A",
                value=MODELS["vLLM"].model_name
            )


            base_url_box = gr.Textbox(
                label="Request URL",
                placeholder="https://api.openai.com/v1",
                value=f"{config.get('base_url', 'https://api.openai.com/v1')}/api_key={config.get('api_key', '')}"
            )


            temperature_slider = gr.Slider(label="Temperature", minimum=0, maximum=1, step=0.1, value=temperature)
       


    def change_temperature(new_temperature):
        global temperature
        temperature = new_temperature


    temperature_slider.change(fn=change_temperature, inputs=temperature_slider)


app.queue().launch(debug=True, share=False)



