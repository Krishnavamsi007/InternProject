from typing import Literal

from pydantic import BaseModel


class ClaimRequest(BaseModel):
    """Everything needed to score one claim. Field names are snake_case and
    match the Gradio form's variable names one-to-one, so the payload built
    in gradio_app.py can be sent to POST /claim with no renaming.
    """

    claim_date: str
    service_date: str
    policy_expiration_date: str
    claim_amount: float
    patient_age: int
    patient_gender: str
    patient_city: str
    patient_state: str
    provider_type: str
    provider_specialty: str
    provider_city: str
    provider_state: str
    diagnosis_code: str
    procedure_code: int
    number_of_procedures: int
    admission_type: str
    discharge_type: str
    length_of_stay_days: int
    service_type: str
    deductible_amount: float
    copay_amount: float
    num_previous_claims_patient: int
    num_previous_claims_provider: int
    provider_patient_distance_miles: float
    claim_submitted_late: bool


class ClaimResponse(BaseModel):
    """Returned by POST /claim right after scoring + saving a new claim."""

    claim_id: int
    prediction: Literal["Fraudulent", "Not Fraudulent"]
    probability: float
    risk_tier: Literal["low", "medium", "high"]


class ClaimLookupResponse(BaseModel):
    """Returned by GET /claim/{claim_id} for a claim that exists."""

    claim_id: int
    prediction: Literal["Fraudulent", "Not Fraudulent"]
    probability: float
    risk_tier: Literal["low", "medium", "high"]
    created_at: str


class ClaimHistoryItem(BaseModel):
    """One row in the GET /claims history list."""

    claim_id: int
    claim_amount: float
    patient_age: int
    provider_type: str
    prediction: Literal["Fraudulent", "Not Fraudulent"]
    probability: float
    risk_tier: Literal["low", "medium", "high"]
    created_at: str
