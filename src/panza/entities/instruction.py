from abc import ABC
from dataclasses import dataclass, field
from typing import List, Optional

from ..llm import ChatHistoryType


@dataclass
class Instruction(ABC):
    instruction: str
    context: Optional[str]
    past_messages: ChatHistoryType = field(default_factory=list)


@dataclass(kw_only=True)
class EmailInstruction(Instruction):
    thread: List[str] = field(default_factory=list)


@dataclass(kw_only=True)
class SnippetInstruction(Instruction):
    pass


@dataclass(kw_only=True)
class SummarizationInstruction(Instruction):
    pass
