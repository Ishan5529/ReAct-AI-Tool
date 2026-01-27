# ReAct-AI-Tool
A powerful **Reasoning and Acting (ReAct) Agent** built with Python and **LangChain**. This tool utilizes **Groq** (openai/gpt-oss-120b / Llama 3 / Qwen) for high-speed inference and connects to external APIs to fetch real-time data (Weather, Web Search) when general knowledge is insufficient.

It features both a **Command Line Interface (CLI)** and a **Gradio Web Interface**.  

## Features

*   **ReAct Logic:** The agent reasons about a query, decides if it needs tools, acts, and then answers.
*   **Tool Integration:**
    *   **Tavily AI:** For advanced web search and current events.
    *   **OpenWeatherMap:** For real-time meteorological data.
    *   **Real-time Clock:** For temporal awareness.
*   **Ambiguity Handling:** The agent is programmed to pause and ask for clarification if user inputs are vague (e.g., missing city names or specific parameters).
*   **Interfaces:**
    *   Terminal-based chat.
    *   Modern web UI using Gradio (Chat history, tool output visualization, model reasoning).  

## Prerequisites

Before running the project, ensure you have the following installed:

*   **Python 3.10+**
*   **Git**

You will also need API Keys for the following services:
1.  **Groq API Key:** [Get it here](https://console.groq.com/)
2.  **Tavily API Key:** [Get it here](https://tavily.com/)
3.  **OpenWeatherMap API Key:** [Get it here](https://openweathermap.org/api)  

## 🛠️ Installation & Setup

Follow these steps to set up the project locally.

### 1. Clone the Repository
Open your terminal and run the following commands:

```bash
git clone https://github.com/Ishan5529/ReAct-AI-Tool.git
cd ReAct-AI-Tool
```

### 2. Create a Virtual Environment (Optional)
It is recommended to use a virtual environment to manage dependencies.  

Windows:

```Bash
python -m venv venv
venv\Scripts\activate
```

MacOS / Linux:

```Bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install the required Python packages from the `requirements.txt` file.

```Bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory to store your API keys.
- Create a file named `.env`
- Paste the following content and replace `values` with your actual keys:

```Ini
GROQ_API_KEY=gsk_...
TAVILY_CLIENT_KEY=tvly-...
OPENWEATHERMAP_API_KEY=...
```

## Usage
You can run the application in two modes.  
### Option A: Command Line Interface (CLI)
Best for quick testing and debugging in the terminal.

```Bash
python app.py
```

Type `exit` to quit the application.  

### Option B: Gradio Web Interface
Provides a user-friendly UI with chat history and visibility into the tool calls (JSON format) and model reasoning.

```Bash
python gradio_app.py
```

Once running, open your browser and navigate to:
http://localhost:7860 or http://127.0.0.1:7860

---

## 📂 Project Structure

```text
ReAct-AI-Tool/
├── src/
│   ├── API/
│   │   ├── weather.py       # Wrapper for OpenWeatherMap
│   │   └── web_search.py    # Wrapper for Tavily Search
│   ├── agent.py             # Agent Setup
│   ├── model.py             # LLM Configuration (Groq)
│   └── tools.py             # LangChain Tool definitions (@tool)
├── app.py                   # Main entry point for CLI
├── gradio_app.py            # Main entry point for Web UI
├── rag_summarize.py         # Agent to summarize previous context
├── requirements.txt         # Project dependencies
└── .env                     # API Keys (Not committed to repo)
```

## How it Works

1.  **User Input:** You ask a question (e.g., "What is the weather in Tokyo compared to London?").
2.  **Reasoning:** The Groq LLM analyzes the prompt. It detects that it doesn't know the live weather.
3.  **Tool Execution:** The agent calls the `weather_search_tool` for Tokyo and then for London.
4.  **Synthesis:** The agent combines the data from the tools into a natural language response.
5.  **Context Update:** Another agent takes the previous context and the current user query and agent output to generate a summarized context for the next agent invokation.