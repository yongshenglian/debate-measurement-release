import attrs
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field

from typing import Any, Optional, Dict


from src.llm_api.llm import LLMClient
from src.llm_api import PromptConfig, Message, LLMConfig
from src.environments.base_env import EnvObservation
from omegaconf import DictConfig


class AgentConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    prompt_config: PromptConfig
    name: Optional[str] = "agent"
    llm_config: LLMConfig
    few_shot_num_samples: Optional[int] = None


# Base agent class,
@attrs.define
class BaseAgent(ABC):
    config: AgentConfig

    # placeholders for parameters that will be initialized from the config
    llm: LLMClient = attrs.field(init=False)
    name: str = attrs.field(init=False)

    # basic memory that is just the list of messages for now
    # TODO: memory should be an abstraction that can handle messages, observations, etc.
    # TODO: all the tracking for past messages,and responses should be here and automatically updated
    # memory = []
    msg_history: list[Message] = attrs.field(factory=list)

    # TODO: add tools, not supported yet

    def __attrs_post_init__(self):
        # just use the agent config to setup the agent specific variables like name, etc.
        self.name = self.config.name

    def _init_llm(self):
        self.llm = LLMClient(config=self.config.llm_config)

    def get_system_prompt(self) -> Message:
        return Message(role="system", content=self.config.prompt_config.system_prompt)

    def build(self):
        """
        The method that uses the config to setup the agents and checks that they are working alright.
        Throws exceptions if something is not working.

        TODO: test via a mock call or something
        """
        self._init_llm()

    def add_to_msg_history(self, message: Message):
        self.msg_history.append(message)

    def _reset_msg_history(self):
        self.msg_history = []
        self.add_to_msg_history(self.get_system_prompt())

    def reset(self):
        self._reset_msg_history()

    @abstractmethod
    async def completion(self, **kwargs):
        """
        Use the system prompt and the user payload to get a completion from the LLM
        """
        pass

    async def instruction_completion(self, payload: str):
        """
        Use the system prompt and the user payload to get a completion from the LLM
        """
        return await self.llm(
            messages=[
                Message(role="system", content=self.config.prompt_config.system_prompt),
                Message(role="user", content=payload),
            ]
        )

    async def chat_completion(self):
        """
        Use the system prompt and the user payload to get a completion from the LLM
        """
        return await self.llm(messages=self.msg_history)

    @abstractmethod
    async def act(self):
        # construt the message
        # make the call to LLM
        # parse the response
        pass

    def print_msg_history(self, color_output: bool = True, indent: int = 2) -> None:
        """
        Print the message history in a readable format.

        Args:
            color_output: Whether to use colored output
            indent: Number of spaces to indent content
        """
        from src.utils.formatting import print_message_history

        print_message_history(
            msg_history=self.msg_history,
            agent_name=self.name,
            color_output=color_output,
            indent=indent,
        )
