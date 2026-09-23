from pydantic import BaseModel, Field, ConfigDict


class PredictionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    age: int = Field(ge=0)
    workclass: str | None = None
    fnlwgt: int = Field(ge=0)
    education: str
    education_num: int = Field(alias="education-num",ge=0)
    marital_status: str = Field(alias="marital-status")
    occupation: str | None = None
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain", ge=0)
    capital_loss: int = Field(alias="capital-loss", ge=0)
    hours_per_week: int = Field(alias="hours-per-week", ge=0)
    native_country: str | None = Field(default=None, alias="native-country")


class PredictionResponse(BaseModel):
    prediction: int
    probability: float