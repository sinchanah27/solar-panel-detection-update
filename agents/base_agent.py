# agents/base_agent.py
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Base class for all agents."""

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def run(self, payload: dict) -> dict:
        pass
