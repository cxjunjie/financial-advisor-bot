import os
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src.advisor.data import download_ohlcv
from src.advisor.features import make_features, FEATURE_COLS
from src.advisor.model import load_model, predict
from src.advisor.explain import llm_explain

MODEL_PATH = os.getenv("MODEL_PATH", "models/logreg_pipeline.pkl")

app = FastAPI(title="Financial Advisor Bot")
pipeline = load_model()

class AdviceResponse(BaseModel):
    ticker: str
    decision: str
    confidence: float
    explanation: str
    meta: dict

@app.get("/advise", response_model=AdviceResponse)
def advise(ticker: str = Query(..., min_length=1, max_length=12)):
    t = ticker.strip().upper()
    try:
        # 1. Download data
        df = download_ohlcv(t)
        feat = make_features(df)
        if feat.empty:
            raise ValueError("Not enough data after feature engineering.")
        
        # 2. Extract latest row
        latest_row = feat.iloc[-1]

        # 3. Predict
        proba, decision = predict(pipeline, latest_row, FEATURE_COLS)

        # 4. Explain with LLM
        text, meta = llm_explain(decision, t, proba, latest_row)
        prose = "\n".join(text.splitlines()[:-1]) if "\n" in text else text

        # 5. Return structured response
        return AdviceResponse(
            ticker=t,
            decision=decision,
            confidence=proba,
            explanation=prose,
            meta=meta
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))