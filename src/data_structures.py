from pydantic import BaseModel
from typing import List, Dict, Literal
    
class TreatmentConfig(BaseModel):
    file_name: str
    treatments_cols: List[str]
    scorer_criteria: List[str] | str
    resolve_scorers_criteria: Literal["scorers", "strengths_and_ILs", "None"] = "None"
    

class ControlConfig(BaseModel):
    file_name: str
    scorer_criteria: List[str] | str

class ExperimentConfig(BaseModel):
    control: ControlConfig | None = None
    treatments: List[TreatmentConfig] | None = None