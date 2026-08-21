import pickle
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

import models
from database import Base, engine, get_db
from feature_engineering import add_engineered_features
from schemas import ClaimHistoryItem, ClaimLookupResponse, ClaimRequest, ClaimResponse

MODEL_PATH = "fraud_detection_model.pkl"

app = FastAPI(
    title="Claims Fraud Detection API",
    description="POST a claim to score it for fraud and store it. "
    "GET a claim back by id.",
    version="1.0",
)

# Create the claims table on startup if it doesn't already exist.
Base.metadata.create_all(bind=engine)

# Load the trained pipeline once, at process start, and keep it in memory.
with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

pipeline = bundle["pipeline"]
feature_columns = bundle["feature_columns"]
model_name = bundle["model_name"]


def _risk_tier(probability: float) -> str:
    if probability >= 0.75:
        return "high"
    if probability >= 0.4:
        return "medium"
    return "low"


def _to_raw_row(claim: ClaimRequest) -> pd.DataFrame:
    """Map the snake_case API schema onto the raw column names the training
    pipeline / feature_engineering.py expect (mirrors the original dataset
    schema)."""
    return pd.DataFrame(
        [
            {
                "Claim_Date": claim.claim_date,
                "Service_Date": claim.service_date,
                "Policy_Expiration_Date": claim.policy_expiration_date,
                "Claim_Amount": claim.claim_amount,
                "Patient_Age": claim.patient_age,
                "Patient_Gender": claim.patient_gender,
                "Patient_City": claim.patient_city,
                "Patient_State": claim.patient_state,
                "Hospital_ID": 0,  # placeholder, dropped by feature engineering
                "Provider_Type": claim.provider_type,
                "Provider_Specialty": claim.provider_specialty,
                "Provider_City": claim.provider_city,
                "Provider_State": claim.provider_state,
                "Diagnosis_Code": claim.diagnosis_code,
                "Procedure_Code": claim.procedure_code,
                "Number_of_Procedures": claim.number_of_procedures,
                "Admission_Type": claim.admission_type,
                "Discharge_Type": claim.discharge_type,
                "Length_of_Stay_Days": claim.length_of_stay_days,
                "Service_Type": claim.service_type,
                "Deductible_Amount": claim.deductible_amount,
                "CoPay_Amount": claim.copay_amount,
                "Number_of_Previous_Claims_Patient": claim.num_previous_claims_patient,
                "Number_of_Previous_Claims_Provider": claim.num_previous_claims_provider,
                "Provider_Patient_Distance_Miles": claim.provider_patient_distance_miles,
                "Claim_Submitted_Late": claim.claim_submitted_late,
            }
        ]
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": model_name}


@app.post("/claim", response_model=ClaimResponse)
def create_claim(claim: ClaimRequest, db: Annotated[Session, Depends(get_db)]):
    """Score a claim for fraud and persist both the claim and the verdict."""
    row = _to_raw_row(claim)
    row_fe = add_engineered_features(row).reindex(columns=feature_columns)

    probability = float(pipeline.predict_proba(row_fe)[0, 1])
    predicted = int(probability >= 0.5)
    tier = _risk_tier(probability)

    db_claim = models.Claim(
        **claim.model_dump(),
        prediction=predicted,
        probability=probability,
        risk_tier=tier,
    )
    db.add(db_claim)
    db.commit()
    db.refresh(db_claim)

    return ClaimResponse(
        claim_id=db_claim.id,
        prediction="Fraudulent" if predicted == 1 else "Not Fraudulent",
        probability=probability,
        risk_tier=tier,
    )


@app.get("/claim/{claim_id}", response_model=ClaimLookupResponse)
def get_claim(claim_id: int, db: Annotated[Session, Depends(get_db)]):
    """Look up a previously scored claim by id."""
    db_claim = db.query(models.Claim).filter(models.Claim.id == claim_id).first()

    if db_claim is None:
        raise HTTPException(status_code=404, detail="No claim found")

    return ClaimLookupResponse(
        claim_id=db_claim.id,
        prediction="Fraudulent" if db_claim.prediction == 1 else "Not Fraudulent",
        probability=db_claim.probability,
        risk_tier=db_claim.risk_tier,
        created_at=db_claim.created_at.isoformat() if db_claim.created_at else "",
    )


@app.get("/claims", response_model=list[ClaimHistoryItem])
def list_claims(db: Annotated[Session, Depends(get_db)], limit: int = 50):
    """Most recent claims first, capped at `limit` (default 50, max 200)."""
    limit = max(1, min(limit, 200))
    db_claims = (
        db.query(models.Claim).order_by(models.Claim.id.desc()).limit(limit).all()
    )

    return [
        ClaimHistoryItem(
            claim_id=c.id,
            claim_amount=c.claim_amount,
            patient_age=c.patient_age,
            provider_type=c.provider_type,
            prediction="Fraudulent" if c.prediction == 1 else "Not Fraudulent",
            probability=c.probability,
            risk_tier=c.risk_tier,
            created_at=c.created_at.isoformat() if c.created_at else "",
        )
        for c in db_claims
    ]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
