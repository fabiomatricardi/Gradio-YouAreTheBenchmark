# YouAreTheBenchmark - LLM Evaluation Suite

A Gradio-based interface for automated LLM evaluation using llama-server.exe API endpoints with human feedback collection.

## Overview

This project transforms your local LLM evaluation workflow into a professional UI with two main tabs:

1. **Server Control**: Start/stop llama-server.exe process with model selection
2. **Benchmark Evaluation**: Run automated prompts with human scoring and export results

The system uses OpenAI's Python client to communicate with llama-server.exe via HTTP API endpoints, providing a clean separation between evaluation logic and inference server.

## Requirements

- Python 3.8+
- llama-server.exe (from llama.cpp project)
- GGUF model files (.gguf)
- Required Python packages:
  - gradio
  - openai
  - pandas
  - openpyxl
  - tiktoken

## Installation

1. **Download llama-server.exe**

   - Get the latest release from https://github.com/ggerganov/llama.cpp
   - Extract and place `llama-server.exe` and all other files in the ZIP archive of the official Binaries in the `llamaCPP/` directory

2. **Obtain GGUF Models**

   - Download Llama 3.2 or other GGUF models from Hugging Face (e.g., bartowski/Llama-3.2-3B-Instruct-GGUF)
   - Place `.gguf` files in the `llamaCPP/models/` directory

3. **Install Python dependencies**

   ```bash
   pip install gradio openai pandas openpyxl tiktoken
   ```

4. **Verify file structure**

   Your directory should look like:

   ```
   youarethebenchmark/
   ├── llamaCPP/
   │   ├── llama-server.exe
   │   ├── ...   
   │   └── models/
   │       ├── Llama-3.2-3B-Instruct-Q5_K_M.gguf
   │       └── ...
   ├── gradio_interface_v2.py
   ├── promptLibv2Qwen.py
   └── README.md
   ```

## Usage Guide

### 1. Launch the Interface

```bash
python gradio_interface_v2.py
```

This will start a web server at `http://127.0.0.1:7860`. Open this URL in your browser.

### 2. Server Control Tab

1. Click "Refresh Models" to load available GGUF files
2. Select a model from the dropdown
3. Click "Start Server" to launch llama-server.exe
   - Server will start with parameters: `-m models/selectedmodel.gguf -c 8192`
   - Status will change to "✅ Running (PID: xxx)"
4. To stop: Click "Stop Server" (graceful shutdown)

> **Note:** The server might take 10-30 seconds to become available after starting. The interface will automatically detect when it's ready.

### 3. Benchmark Evaluation Tab

1. Click "Refresh Models" to load available models
2. Select the same model you started in Server Control
3. Click "Start Evaluation"
4. The system will:
   - Process all 14 NLP tasks sequentially
   - Show progress bar with current task description
   - Send requests to llama-server.exe via OpenAI API
   - Automatically collect timing metrics
   - Wait for your feedback on each task
   - Update the results table in real-time
   - Save results to `logs/benchmark_results_XXXXXXX.xlsx`
   - Save a detailed session log to `logs/session_log_XXXXXXX.txt`

### 4. Human Feedback

The evaluation includes interactive human feedback for each task:

**For each task:**
1. System displays the task description
2. System shows the prompt sent to the model
3. LLM generates a response and shows timing metrics
4. **View the response** in the left panel
5. **Rate the quality** on a scale of 0-5 (where 5 is excellent)
6. Add **optional notes** about the response
7. Click "Submit & Next Task" to proceed

**Rating Scale:**
- **0** = Gibberish or completely off-topic
- **1** = Poor quality, ignores instructions
- **2** = Barely adequate, follows some instructions
- **3** = Acceptable, follows instructions with minor issues
- **4** = Good quality, follows instructions accurately
- **5** = Perfect, follows instructions precisely with excellent content

The system pauses at each task until you provide feedback, ensuring thorough evaluation of every response.

### 5. Results and Scoring

After completing all tasks:

- **Results table** will display all evaluations with:
  - Task name
  - Prompt tokens count
  - Response tokens count
  - Latency in seconds
  - Quality rating
  - User notes

- **Excel export** (`logs/benchmark_results_XXXXXXX.xlsx`) contains:
  - task, prompt, response (full text)
  - prompt_tokens, response_tokens
  - latency, rating, notes

- **Session log** (`logs/session_log_XXXXXXX.txt`) contains:
  - Complete task-by-task log with all prompts and responses
  - Your ratings and notes for each task
  - Performance metrics
  - **Final score**: X/70 (where 70 is maximum possible score = 14 tasks × 5 points)

## Features

- 🔁 Automatic model discovery in `llamaCPP/models/`
- 🚀 Server process management with PID tracking
- ⏱️ Performance metrics: latency, token counts
- 📊 Automated results export to Excel
- 📜 Detailed session logs with full prompt/response history
- 🎯 **Total score calculation** (maximum 70 points)
- 🌐 Gradio web interface with Soft theme
- 📐 **Responsive layout** with organized columns
- 🧪 Standardized NLP task catalog (14 evaluation tasks)
- 📝 Interactive human feedback for each task (0-5 rating + notes)
- ⏸️ Sequential processing with user control between tasks
- 📈 **Real-time progress tracking** with visible progress bar
- 📋 **Real-time results table** updating after each evaluation

## Interface Layout

The Benchmark Evaluation tab uses a modern two-column layout:

- **Left Column (60%)**: Task display, prompt, and LLM response
- **Right Column (40%)**: Feedback form with rating slider and notes
- **Progress Section**: Progress text bar and visual progress slider
- **Results Table**: Live-updating table of all completed evaluations

## Data Export

After evaluation completes, results are saved to two files:

### 1. Excel File
`logs/benchmark_results_XXXXXXX.xlsx`

Contains columns:
- task: NLP task description
- prompt: Full prompt text sent to the model
- response: Full model response
- prompt_tokens: Number of input tokens
- response_tokens: Number of output tokens
- latency: Time in seconds for full generation
- rating: Human quality score (0-5)
- notes: User comments about the response

### 2. Session Log
`logs/session_log_XXXXXXX.txt`

Contains detailed entries for each task including:
- Task number and description
- Full prompt
- Full assistant response
- Rating and notes
- Latency and token counts
- Final summary with total score and percentage

## Troubleshooting

### Server won't start

- Verify `llama-server.exe` exists in `llamaCPP/`
- Check that `.gguf` files are in `llamaCPP/models/`
- Ensure no firewall is blocking port 8080
- Try running `llama-server.exe -m models/Llama-3.2-3B-Instruct-Q5_K_M.gguf -c 8192` manually in cmd

### API connection errors

- Wait 20-30 seconds after starting server before initiating evaluation
- Verify server is listening on `http://localhost:8080/v1`
- Try accessing `http://localhost:8080/v1/models` in your browser

### Missing Python packages

```bash
pip install gradio openai pandas openpyxl
```

### Results table not updating

- Ensure you're submitting feedback after each task
- Check that the results table is not hidden (scroll down in the interface)
- Refresh the page if necessary (this won't lose current evaluations)

## Technical Details

- **Theme**: Gradio Soft theme for a modern, clean interface
- **API Compatibility**: Uses OpenAI-compatible API via llama-server.exe
- **Temperature**: Set to 0.15 for consistent, deterministic outputs
- **Max Tokens**: 1500 tokens per response
- **Context Size**: 8192 tokens
- **Score Calculation**: Total score = sum of all task ratings (max = 14 × 5 = 70)

## Future Improvements

1. **Real-time streaming**: Show LLM responses word-by-word during generation
2. **Advanced metrics**: Tokenization analysis, perplexity calculations
3. **Multi-user support**: Persistent storage of evaluations in a database
4. **Dashboard**: Visualizations of rating distributions and score trends
5. **Batch comparison**: Compare multiple models side-by-side
6. **Export formats**: CSV, JSON, and PDF reporting options
7. **Custom task sets**: Allow users to define their own evaluation tasks
8. **Model comparison mode**: Directly compare two models on the same tasks

## License

MIT License

## Credits

- Inspired by https://github.com/ggerganov/llama.cpp
- Base concepts from promptLibv2Qwen.py by Fabio Matricardi

> This UI is designed for researchers and developers evaluating LLM quality in controlled environments. The human feedback component provides systematic quality assessment for each task, with comprehensive logging and scoring capabilities.

> created with <img src='https://github.com/anomalyco/opencode/raw/dev/packages/console/app/src/asset/logo-ornate-light.svg' height=20> and free Zen GLM-4.7 model. Entire coding session [documented here](https://github.com/fabiomatricardi/Gradio-YouAreTheBenchmark/raw/main/session-ses_4251.md) 

---

## Testing the Model

You can test if the model is running from PowerShell terminal by running the following `curl` command:

```bash
curl.exe http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\`"model\`": \`"gpt-3.5-turbo\`", \`"messages\`": [{\`"role\`": \`"user\`", \`"content\`": \`"Hello!\`"}]}"
```

