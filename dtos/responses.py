from typing import List
from pydantic import BaseModel

class PredictResponse(BaseModel):
    ratings: List[float]
