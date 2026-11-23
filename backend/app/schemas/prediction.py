from pydantic import BaseModel, ConfigDict, Field


class StudentFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    gender: str = Field(alias="Gender")
    age: float = Field(alias="Age")
    city: str = Field(alias="City")
    profession: str = Field(alias="Profession")
    academic_pressure: float = Field(alias="Academic Pressure")
    work_pressure: float = Field(alias="Work Pressure")
    cgpa: float = Field(alias="CGPA")
    study_satisfaction: float = Field(alias="Study Satisfaction")
    job_satisfaction: float = Field(alias="Job Satisfaction")
    sleep_duration: str = Field(alias="Sleep Duration")
    dietary_habits: str = Field(alias="Dietary Habits")
    degree: str = Field(alias="Degree")
    work_study_hours: float = Field(alias="Work/Study Hours")
    financial_stress: float = Field(alias="Financial Stress")
    family_history: str = Field(alias="Family History of Mental Illness")

    def to_feature_dict(self) -> dict[str, object]:
        return self.model_dump(by_alias=True)


class PredictionResponse(BaseModel):
    probability_yes: float
    probability_no: float
    dataset_label_estimate: str
    threshold: float
    model_version: str
    explanation_available: bool
    disclaimer: str


class ExplanationRequest(StudentFeatures):
    pass


class ExplanationResponse(BaseModel):
    available: bool
    message: str
    base_value: float | None = None
    model_output: float | None = None
    output_scale: str | None = None
    additivity_error: float | None = None
    contributions: list[dict[str, float | str]] = Field(default_factory=list)
