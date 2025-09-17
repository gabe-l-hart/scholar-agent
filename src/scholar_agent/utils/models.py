"""
Factory for constructing langchain models
"""

from langchain_core.language_models import BaseChatModel

# Third Party
from langchain_ollama import ChatOllama

# Local
from scholar_agent.types import ModelProvider
from scholar_agent.utils.factory import FactoryConstructible, ImportableFactory


class ModelFactory(ImportableFactory):
    def __init__(self):
        super().__init__("model")

    def register(self, name_: str, model_type: type[BaseChatModel]):
        class ConstructionWrapper(FactoryConstructible):
            name = name_

            def __init__(self, config, instance_name: str):
                self._inst = model_type(**config)

            def __getattr__(self, name):
                return getattr(self._inst, name)

        super().register(ConstructionWrapper)


model_factory = ModelFactory()
model_factory.register(ModelProvider.OLLAMA.value, ChatOllama)
