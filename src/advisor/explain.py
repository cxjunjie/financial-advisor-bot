import json
import pandas as pd
import ollama

def build_prompt(decision: str, ticker: str, conf: float, latest: pd.Series) -> str:
    rsi = float(latest["rsi_14"]); sma10 = float(latest["sma_10"]); sma50 = float(latest["sma_50"])
    return f"""
You are a careful financial explainer. Explain to a beginner why the model recommends **{decision}** for {ticker}.

Facts:
- Confidence (0-1): {conf:.2f}
- RSI14: {rsi:.1f}
- SMA10: {sma10:.2f}
- SMA50: {sma50:.2f}

Rules (must follow):
- If RSI14 < 30 → "oversold". If RSI14 > 70 → "overbought". Else "RSI is neutral".
- If SMA10 > SMA50 → "short-term above long-term (bullish)". If < → "bearish". If equal → "neutral".
- ≤120 words. Start with one-line summary, then 2 - 3 bullets.
- End with: "This is educational, not financial advice."

Also output a final JSON line with keys:
decision, confidence, reasons (list of strings), disclaimer.
The JSON must be the last line only.
""".strip()

def llm_explain(decision: str, ticker: str, conf: float, latest: pd.Series) -> tuple[str, dict]:
    prompt = build_prompt(decision, ticker, conf, latest)
    resp = ollama.chat(
        model="llama3.2:latest",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1}
    )
    text = resp["message"]["content"].strip()
    last = text.splitlines()[-1]
    try:
        meta = json.loads(last)
    except Exception:
        meta = {
            "decision": decision,
            "confidence": round(conf, 3),
            "reasons": [f"RSI14={latest['rsi_14']:.1f}",
                        f"SMA10={latest['sma_10']:.2f}",
                        f"SMA50={latest['sma_50']:.2f}"],
            "disclaimer": "This is educational, not financial advice."
        }
    return text, meta
