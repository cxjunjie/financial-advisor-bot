# Financial Advisor Bot

A simple AI-powered financial chatbot that provides BUY/SELL recommendations with beginner friendly explanations. It combines a machine learning model (Logistic Regression pipeline) with a Large Language Model (LLM) via Ollama to explain stock recommendations in natural language.

# Features
- Downloads live OHLCV data using Yahoo Finance (yfinance).

- Computes technical indicators (RSI, SMA10, SMA50).

- Runs a fine-tuned Logistic Regression pipeline for BUY/SELL predictions.

- Generates beginner-friendly explanations using Ollama (llama 3.2).

- Exposes a REST API via FastAPI (/advise?ticker="tickername").

- JSON responses include:
  - Decision (BUY/SELL)
  - Confidence score
  - Model reasoning
  - LLM-generated natural language explanation
  - Disclaimer
 
# Project Structure

financial-advisor-bot/
  - models/                  # Saved LogReg model (logreg_pipeline.pkl)
    - logreg_pipeline.pkl
  - src/advisor/
    - data.py              # Download OHLCV data
    - features.py          # Feature engineering (RSI, SMA10, SMA50)
    - model.py             # Load model + make predictions
    - explain.py           # LLM integration for explanations
  - app.py               # FastAPI app (main entry point)
  - requirements.txt         # Python dependencies
  - README.md                # Project documentation

# Installation

1. Clone the repo
2. Create a virtual environment
3. Install the dependencies
4. Install Ollama
  - ollama pull llama3.2:latest

# Running the app
1. Run in terminal: uvicorn src.advisor.app:app --reload
2. Open browser and type in: http://127.0.0.1:8000/docs

