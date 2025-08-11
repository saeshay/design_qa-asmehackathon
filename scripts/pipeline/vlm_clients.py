from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class ModelOutput:
    text: str
    confidence: float = 0.5


class BaseClient:
    def generate(self, prompt: str, image_path: Optional[str] = None) -> ModelOutput:
        raise NotImplementedError


class MockClient(BaseClient):
    def __init__(self, name: str):
        self.name = name

    def generate(self, prompt: str, image_path: Optional[str] = None) -> ModelOutput:
        # Simple echo fallback for offline use
        if image_path:
            return ModelOutput(text="Explanation: INSUFFICIENT\nAnswer: INSUFFICIENT", confidence=0.1)
        return ModelOutput(text="INSUFFICIENT", confidence=0.1)


class Ensemble:
    def __init__(self, clients: List[BaseClient]):
        self.clients = clients

    def query(self, prompt: str, image_path: Optional[str] = None) -> ModelOutput:
        outputs = [c.generate(prompt, image_path) for c in self.clients]
        # Majority vote over normalized forms where applicable will be handled by task heads
        # Here we just pick the highest confidence as a default tie-breaker
        best = max(outputs, key=lambda x: x.confidence)
        return best