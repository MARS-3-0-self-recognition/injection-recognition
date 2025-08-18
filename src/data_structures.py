from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Literal
from inspect_ai.model import Model

class TreatmentConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: str | Model | None = None
    file_name: str | None = None
    treatments_cols: List[str]
    scorer_criteria: tuple[str, str] | tuple[str, str, str]
    resolve_scorers_criteria: Literal["scorers", "strengths_and_ILs", "None"] = "None"

class ControlConfig(BaseModel):
    file_name: str
    scorer_criteria: tuple[str, str]

class ExperimentConfig(BaseModel):
    control: ControlConfig | None = None
    treatments: List[TreatmentConfig] | None = None