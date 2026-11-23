import attrs
from typing import List, Optional, Literal

from src.agents.base_agent import BaseAgent
from src.llm_api.base_llm_abstractions import Message


@attrs.define
class BasicDebateAgent(BaseAgent):
    summarize: bool = attrs.field(default=False)
    use_all_history: bool = attrs.field(default=False)
    num_agents: int = attrs.field(default=None)
    total_cost: float = attrs.field(default=0.0)

    domain: Literal["mmlu", "math", "commonsense_qa", "gsm8k"] = attrs.field(
        default=None
    )

    # NOTE: hardcoded for now
    instruction: str = attrs.field(default=None)
    answer_history: List[str] = attrs.field(factory=list)
    debug: bool = attrs.field(default=False)

    def reset(self):
        """
        Reset the agent for a new debate.
        """
        # debate specific history
        self.instruction = None
        self.answer_history = []
        self.total_cost = 0

        # base agent history
        self._reset_msg_history()

    def set_instruction(self, instruction: str):
        """
        Set the instruction for the agent.
        """
        self.instruction = instruction
        self.add_to_msg_history(Message(role="user", content=self.instruction))

    async def completion(self) -> str:
        """
        Use the LLM while treating this as Instruction following task
        """
        response = await self.chat_completion()

        # Extract the completion text from the response
        response_text = response.completion
        self.total_cost += response.cost

        # Add the response to the message history
        self.add_to_msg_history(Message(role="assistant", content=response_text))

        return response_text

    def get_total_cost(self) -> float:
        """
        Get the total cost incurred by this agent
        """
        return self.total_cost

    async def act(self):
        raise NotImplementedError(
            "MMLUDebateAgent does not implement act because there is no environment"
        )

    async def summarize_other_agents(self, agent_responses: List[str]):
        """
        Summarize the responses from other agents, and return the summary
        via the instruction completion API.
        """

        summarization_message = self.config.prompt_config.partials[
            "summarization_message_initial"
        ]

        for agent_response in agent_responses:
            summarization_message += "\n\n One agent response: ```{}```".format(
                agent_response
            )

        summarization_message += (
            "\n\n " + self.config.prompt_config.partials["summarization_message_final"]
        )

        response = await self.instruction_completion(payload=summarization_message)
        response_text = response.completion
        return response_text

    async def generate_answer(self):
        answer = await self.completion()
        self.answer_history.append(answer)

        # If debug is true, print the message history
        if self.debug:
            self.print_msg_history()

        return answer

    async def add_discussion_with_other_agents_in_context(
        self,
        agent_responses: List[str],
        summarize: bool = False,
        additional_context: Optional[str] = None,
    ) -> None:
        # constuct other agent responses by summarizing if needed
        if summarize:
            other_agent_responses = await self.summarize_other_agents(agent_responses)
        else:
            other_agent_responses = ""
            for agent_response in agent_responses:
                other_agent_responses += "\n\n One agent response: ```{}```".format(
                    agent_response
                )

        if self.domain == "mmlu":
            payload = self.config.prompt_config.partials["discussion_message"].format(
                AGENT_RESPONSES=other_agent_responses
            )
        elif self.domain == "math":
            payload = self.config.prompt_config.partials["discussion_message"].format(
                AGENT_RESPONSES=other_agent_responses,
                QUESTION_PROMPT=additional_context,
            )
        elif self.domain == "commonsense_qa":
            payload = self.config.prompt_config.partials["discussion_message"].format(
                AGENT_RESPONSES=other_agent_responses
            )
        elif self.domain == "gsm8k":
            payload = self.config.prompt_config.partials["discussion_message"].format(
                AGENT_RESPONSES=other_agent_responses,
                QUESTION_PROMPT=additional_context,
            )
        else:
            raise ValueError(
                f"Domain {self.domain} not supported for Agent: {self.name}"
            )

        # self.running_context += "\n\n" + payload
        self.add_to_msg_history(Message(role="user", content=payload))

    def latest_response(self) -> str:
        """
        Get the latest response from the agent.
        """
        return self.answer_history[-1]

    def latest_query(self) -> str:
        """
        Get the latest query from the agent.
        """
        for message in reversed(self.msg_history):
            if message.role == "user":
                return message.content
        return ""
