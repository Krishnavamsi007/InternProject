from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String

from database import Base


class Claim(Base):
    """One row per submitted claim: the raw claim fields the user entered,
    plus the model's verdict for that claim. Storing both together keeps
    GET /claim/{claim_id} a single simple lookup.
    """

    __tablename__ = "claims"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # --- raw claim fields (mirrors ClaimRequest in schemas.py) -------------
    claim_date = Column(String(10))
    service_date = Column(String(10))
    policy_expiration_date = Column(String(10))
    claim_amount = Column(Float)
    patient_age = Column(Integer)
    patient_gender = Column(String(20))
    patient_city = Column(String(100))
    patient_state = Column(String(10))
    provider_type = Column(String(50))
    provider_specialty = Column(String(100))
    provider_city = Column(String(100))
    provider_state = Column(String(10))
    diagnosis_code = Column(String(20))
    procedure_code = Column(Integer)
    number_of_procedures = Column(Integer)
    admission_type = Column(String(50))
    discharge_type = Column(String(50))
    length_of_stay_days = Column(Integer)
    service_type = Column(String(50))
    deductible_amount = Column(Float)
    copay_amount = Column(Float)
    num_previous_claims_patient = Column(Integer)
    num_previous_claims_provider = Column(Integer)
    provider_patient_distance_miles = Column(Float)
    claim_submitted_late = Column(Boolean)

    # --- model verdict, filled in at prediction time ------------------------
    prediction = Column(Integer)  # 0 = not fraudulent, 1 = fraudulent
    probability = Column(Float)
    risk_tier = Column(String(10))  # "low" | "medium" | "high"

    created_at = Column(DateTime, default=datetime.utcnow)
