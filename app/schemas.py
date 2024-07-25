from pydantic import BaseModel

class GeneratePayload(BaseModel):
    topic: str
    
class AnalyzePayload(BaseModel):
    content: str