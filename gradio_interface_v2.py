import os
import subprocess
import time
import pandas as pd
import gradio as gr
from openai import OpenAI
from promptLibv2Qwen import createCatalog
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize global variables
server_process: subprocess.Popen | None = None
server_pid: int | None = None
server_running = False

# Evaluation state
current_task_index = 0
evaluation_results = []
current_tasks = []
current_response = ""
current_prompt = ""
current_latency = 0

def get_gguf_models():
    """Scan models directory and return list of GGUF files"""
    models_dir = os.path.join("llamaCPP", "models")
    if not os.path.exists(models_dir):
        return []
    return [f for f in os.listdir(models_dir) if f.endswith('.gguf')]

def start_server(model_name):
    """Start llama-server.exe with selected model"""
    global server_process, server_pid, server_running
    
    if server_running:
        return "Server already running. Stop it first."
    
    model_path = os.path.join("llamaCPP", "models", model_name)
    if not os.path.exists(model_path):
        return f"Model file not found: {model_path}"
    
    try:
        # Command: llama-server.exe -m models/selectedmodel.gguf -c 8192
        cmd = ["llamaCPP\\llama-server.exe", "-m", model_path, "-c", "8192"]
        server_process = subprocess.Popen(
            cmd,
            cwd="E:\\youarethebenchmark",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        server_pid = server_process.pid
        server_running = True
        
        # Wait for server to be ready (check API endpoint)
        client = OpenAI(base_url="http://localhost:8080/v1", api_key="no-key-needed")
        
        # Poll until server is ready
        max_attempts = 30
        for i in range(max_attempts):
            try:
                models = client.models.list()
                if models.data:
                    logger.info("Server is ready")
                    return f"Server started successfully (PID: {server_pid})"
            except Exception:
                pass
            time.sleep(1)
        
        return f"Server started (PID: {server_pid}), but API not responding"
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        return f"Failed to start server: {str(e)}"

def stop_server():
    """Stop the llama-server.exe process"""
    global server_process, server_pid, server_running
    
    if not server_running:
        return "Server is not running."
    
    try:
        # First try graceful shutdown
        if server_process is not None:
            server_process.terminate()
            server_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        # Force kill if graceful shutdown fails
        if server_process is not None:
            server_process.kill()
            server_process.wait()
    except Exception as e:
        logger.error(f"Error stopping server: {e}")
    finally:
        server_process = None
        server_pid = None
        server_running = False
        
    return "Server stopped."

def check_server_status():
    """Check if server is running"""
    if server_running:
        return f"✅ Running (PID: {server_pid})"
    else:
        return "❌ Stopped"

def initialize_evaluation(model_name):
    """Initialize evaluation session"""
    global current_task_index, evaluation_results, current_tasks, current_response, current_prompt, current_latency
    
    if not server_running:
        return "❌ Server is not running. Please start the server first.", "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=0), gr.update(value=""), gr.update(visible=False), gr.update(visible=False), get_results_table()
    
    # Load tasks
    tasks = createCatalog()
    current_tasks = tasks
    current_task_index = 0
    evaluation_results = []
    
    # Process first task
    return process_next_task(model_name)

def process_next_task(model_name):
    """Process the next task in the queue"""
    global current_task_index, current_response, current_prompt, current_latency
    
    if current_task_index >= len(current_tasks):
        return "✅ All tasks completed!", "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=0), gr.update(value=""), gr.update(visible=False), gr.update(visible=False), get_results_table()
    
    task = current_tasks[current_task_index]
    task_desc = task['task']
    current_prompt = task['prompt']
    
    task_info = f"📋 Task {current_task_index + 1}/{len(current_tasks)}: {task_desc}"
    prompt_info = current_prompt
    
    # Send request to server
    client = OpenAI(base_url="http://localhost:8080/v1", api_key="no-key-needed")
    
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": current_prompt}],
            temperature=0.15,
            max_tokens=1500,
            stream=False
        )
        
        current_response = response.choices[0].message.content
        end_time = time.time()
        current_latency = end_time - start_time
        
        progress_percent = int((current_task_index / len(current_tasks)) * 100)
        
        return task_info, task_info, gr.update(value=current_prompt, visible=True), gr.update(value=current_response, visible=True), gr.update(value=f"⏱️ Latency: {current_latency:.2f}s", visible=True), gr.update(visible=True), gr.update(value=0), gr.update(value=""), gr.update(value=progress_percent), gr.update(visible=False), get_results_table()

    except Exception as e:
        logger.error(f"Error processing task: {e}")
        error_msg = f"❌ Error: {str(e)}"
        return error_msg, task_info, gr.update(value=current_prompt, visible=False), gr.update(value=error_msg, visible=False), gr.update(value="", visible=False), gr.update(visible=False), gr.update(value=0), gr.update(value=""), gr.update(visible=False), gr.update(visible=False), get_results_table()

def submit_feedback(rating, notes, model_name):
    """Submit user feedback and move to next task"""
    global current_task_index, evaluation_results, current_response, current_prompt, current_latency
    
    # Store current task results
    task = current_tasks[current_task_index]
    
    # Count tokens
    prompt_tokens = len(current_prompt.split())
    response_tokens = len(current_response.split())
    
    result = {
        'task': task['task'],
        'prompt': current_prompt,
        'response': current_response,
        'prompt_tokens': prompt_tokens,
        'response_tokens': response_tokens,
        'latency': current_latency,
        'rating': rating,
        'notes': notes
    }
    evaluation_results.append(result)
    
    # Move to next task
    current_task_index += 1
    
    # Update results table for display
    results_df = get_results_table()
    
    # Check if all tasks are done
    if current_task_index >= len(current_tasks):
        # Export results
        total_score, max_score = export_results()
        final_msg = f"✅ All tasks completed! Results saved to Excel and log file.\n📊 Final Score: {total_score}/{max_score} ({(total_score/max_score)*100:.1f}%)"
        return final_msg, "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=0), gr.update(value=""), gr.update(value=100), gr.update(value=final_msg, visible=True), results_df
    else:
        # Process next task
        next_results = list(process_next_task(model_name))
        next_results.append(results_df)
        return tuple(next_results)

def export_results():
    """Export results to Excel and create session log"""
    global evaluation_results
    
    # Create log directory if needed
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = int(time.time())
    
    # Create session log file
    log_path = os.path.join(log_dir, f"session_log_{timestamp}.txt")
    total_score = 0
    max_score = len(evaluation_results) * 5
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("LLM BENCHMARK EVALUATION SESSION LOG\n")
        f.write("="*80 + "\n\n")
        
        for idx, result in enumerate(evaluation_results, 1):
            f.write(f"Task {idx}: {result['task']}\n")
            f.write("-"*80 + "\n")
            f.write(f"PROMPT:\n{result['prompt']}\n\n")
            f.write(f"ASSISTANT RESPONSE:\n{result['response']}\n\n")
            f.write(f"Rating: {result['rating']}/5\n")
            f.write(f"Notes: {result['notes']}\n")
            f.write(f"Latency: {result['latency']:.2f}s\n")
            f.write(f"Tokens: Prompt={result['prompt_tokens']}, Response={result['response_tokens']}\n")
            f.write("="*80 + "\n\n")
            total_score += result['rating']
        
        f.write(f"\n{'='*80}\n")
        f.write(f"FINAL SCORE: {total_score}/{max_score} ({(total_score/max_score)*100:.1f}%)\n")
        f.write("="*80 + "\n")
    
    logger.info(f"Session log saved to {log_path}")
    
    # Create DataFrame for Excel
    df = pd.DataFrame(evaluation_results)
    
    # Save to Excel
    excel_path = os.path.join(log_dir, f"benchmark_results_{timestamp}.xlsx")
    df.to_excel(excel_path, index=False)
    
    logger.info(f"Results exported to {excel_path}")
    
    return total_score, max_score

def get_results_table():
    """Get current results as DataFrame"""
    global evaluation_results
    
    if not evaluation_results:
        return pd.DataFrame(columns=['task', 'prompt_tokens', 'response_tokens', 'latency', 'rating', 'notes'])
    
    df = pd.DataFrame(evaluation_results)
    df = df[['task', 'prompt_tokens', 'response_tokens', 'latency', 'rating', 'notes']]
    return df

# Create the Gradio interface
with gr.Blocks(title="YouAreTheBenchmark - LLM Evaluation Suite",theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 YouAreTheBenchmark - LLM Evaluation Suite")
    
    # Tab 1: Server Control
    with gr.Tab("🖥️ Server Control"):
        gr.Markdown("## Control the LlamaCpp Server")
        
        # Model selection
        model_dropdown = gr.Dropdown(
            choices=get_gguf_models(), 
            label="Select GGUF Model",
            interactive=True
        )
        
        # Refresh button for models
        refresh_btn = gr.Button("🔄 Refresh Models")
        refresh_btn.click(
            fn=get_gguf_models,
            outputs=model_dropdown
        )
        
        # Server controls
        status_display = gr.Textbox(
            value="❌ Stopped",
            label="Server Status",
            interactive=False
        )
        
        with gr.Row():
            start_btn = gr.Button("▶️ Start Server", variant="primary")
            stop_btn = gr.Button("⏹️ Stop Server", variant="stop")
        
        # Status updates
        start_btn.click(
            fn=start_server,
            inputs=model_dropdown,
            outputs=status_display
        )
        
        stop_btn.click(
            fn=stop_server,
            outputs=status_display
        )
        
        # Periodically update status
        demo.load(fn=check_server_status, outputs=status_display)
        demo.load(fn=get_gguf_models, outputs=model_dropdown)
    
    # Tab 2: Benchmark Evaluation
    with gr.Tab("📊 Benchmark Evaluation"):
        gr.Markdown("## Automated Prompt Benchmarking with Human Feedback")
        with gr.Row():
            with gr.Column(scale=3):        
                # Model selection for benchmark
                benchmark_model_dropdown = gr.Dropdown(
                    choices=get_gguf_models(), 
                    label="Select Model for Benchmark",
                    interactive=True
                )
            with gr.Column(scale=2):           
                # Start evaluation button
                start_eval_btn = gr.Button("🚀 Start Evaluation", variant="primary")
        
        # Progress tracking
        progress_info = gr.Textbox(label="Progress", interactive=False)
        progress_bar = gr.Slider(minimum=0, maximum=100, value=0, label="Progress", visible=True, interactive=False)
        
        # Task display
        with gr.Row():
            with gr.Column(scale=3):
                task_display = gr.Markdown(label="Current Task", visible=False)
                prompt_display = gr.Textbox(label="Prompt", visible=False, interactive=False, lines=5)
                response_display = gr.Textbox(label="LLM Response", visible=False, interactive=False, lines=10)
                latency_display = gr.Textbox(label="Performance", visible=False, interactive=False)
            with gr.Column(scale=2):
                # Feedback section (hidden initially)
                with gr.Group(visible=False) as feedback_group:
                    gr.Markdown("### 📝 Your Feedback")
                    rating_slider = gr.Slider(
                        minimum=0,
                        maximum=5,
                        step=0.5,
                        value=0,
                        label="Quality Rating (0-5)",
                        info="0=Bad, 5=Excellent"
                    )
                    notes_input = gr.Textbox(
                        label="Notes (optional)",
                        placeholder="Add any comments about this response...",
                        lines=3
                    )
                    submit_feedback_btn = gr.Button("✅ Submit & Next Task", variant="primary")
        
        # Completion message
        completion_message = gr.Markdown(visible=False)
        
        # Results table
        results_table = gr.Dataframe(
            headers=["task", "prompt_tokens", "response_tokens", "latency", "rating", "notes"],
            label="Evaluation Results",
            interactive=False
        )
        
        # Refresh models button
        refresh_benchmark_btn = gr.Button("🔄 Refresh Models")
        refresh_benchmark_btn.click(
            fn=get_gguf_models,
            outputs=benchmark_model_dropdown
        )
        
        # Start evaluation
        start_eval_btn.click(
            fn=initialize_evaluation,
            inputs=benchmark_model_dropdown,
            outputs=[
                progress_info,
                task_display,
                prompt_display,
                response_display,
                latency_display,
                feedback_group,
                rating_slider,
                notes_input,
                progress_bar,
                completion_message,
                results_table
            ]
        )
        
        # Submit feedback
        submit_feedback_btn.click(
            fn=submit_feedback,
            inputs=[rating_slider, notes_input, benchmark_model_dropdown],
            outputs=[
                progress_info,
                task_display,
                prompt_display,
                response_display,
                latency_display,
                feedback_group,
                rating_slider,
                notes_input,
                progress_bar,
                completion_message,
                results_table
            ]
        )
        
        # Auto-refresh models in benchmark tab
        demo.load(fn=get_gguf_models, outputs=benchmark_model_dropdown)
        demo.load(fn=get_results_table, outputs=results_table)

# Launch the interface
if __name__ == "__main__":
    demo.launch()