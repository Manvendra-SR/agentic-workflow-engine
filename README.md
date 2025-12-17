# Hierarchical Agentic AI System

## Overview
This project implements a **hierarchical multi-agent system** using **LangGraph**, **LangChain**, and **Gemini LLMs**, wrapped inside an interactive **Chainlit UI**.

The system is designed to demonstrate **agent orchestration**, **tool-augmented reasoning**, and **clean separation between agent internals and user-facing output**.

Unlike simple chatbot projects, this architecture models real-world agent systems with:
- Supervisors
- Specialized worker teams
- Controlled execution flow
- Async-safe UI integration

---

## Architecture

### High-level Flow
```
User Query
   ↓
Top-Level Supervisor
   ├── Research Team (Search + Web Scraping)
   └── Writing Team (Docs, Notes, Charts)
   ↓
Final Answer (User-facing)
```

### Key Concepts Used
- **LangGraph StateGraph** for agent control flow
- **Supervisor–Worker hierarchy**
- **Tool-based delegation**
- **Async-safe execution**
- **UI trace vs final output separation**

---

## Tech Stack

- **Python 3.10+**
- **LangGraph**
- **LangChain**
- **Gemini (Google Generative AI)**
- **Chainlit**
- **Asyncio**
- **dotenv**

---

## Features

- Hierarchical agent orchestration (Supervisor → Teams → Tools)
- Modular graph construction
- Tool-augmented reasoning (search, scraping, Python REPL)
- Async-safe rate limiting
- Chainlit UI with:
  - Clean final answers
  - Collapsible internal execution trace
- Side-effect–free imports using factory patterns

---

## Project Structure

```
project/
│
├── app.py                 # Chainlit entrypoint
├── src/
│   ├── helper.py          # LLM, tools, and node definitions
│   ├── prompt.py          # Prompt templates
│   ├── graphs.py          # Graph factory functions (recommended)
│   └── __init__.py
│
├── setup.py
├── chainlit.md
└── README.md
```

---

## Environment Setup

### 1. Create and activate virtual environment
```bash
conda create -n agentic python=3.10
conda activate agentic
```

### 2. Install dependencies
```bash
pip install -e .
pip install chainlit
```

### 3. Set environment variables
```bash
export GOOGLE_API_KEY=your_key_here
# optional
export MISTRAL_API_KEY=your_key_here
```

---

## Running the App

```bash
chainlit run app.py -w
```

Open in browser:
```
http://localhost:8000
```

---

## UI Behavior

- **Chat Window** → Only final agent output
- **Execution Trace Panel** → Full agent reasoning and routing
- **No internal state leakage to user**

This mirrors production LLM systems like ChatGPT.

---

## Why This Project Matters

This project goes beyond basic LLM usage and demonstrates:

- Real agent orchestration patterns
- Explicit control flow (not hidden chains)
- Async correctness
- Debuggable, inspectable agent behavior
- Production-oriented UI separation

---

## Known Limitations

- Free-tier Gemini rate limits
- No persistent memory (can be added)
- No production deployment config

These are **infrastructure constraints**, not design flaws.

---

## Future Improvements

- Persistent memory (vector store)
- Agent-level caching
- Configurable rate limiter
- Deployment-ready Docker setup
- Multi-user session isolation

---

## Author

**Manvendra Singh**  
B.Tech Mechanical Engineering, IIT Indore  
Interested in AI systems, agentic architectures, and ML platforms

---

## License

MIT
