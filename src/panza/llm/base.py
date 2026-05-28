from abc import ABC, abstractmethod
import glob
import os
from typing import Dict, Iterator, List, Literal

MessageType = Dict[Literal["role", "content"], str]
ChatHistoryType = List[MessageType]


class LLM(ABC):
    def __init__(self, name: str, sampling: Dict):
        self.name = name
        self.sampling_parameters = sampling


    def set_latest_model(self, checkpoint_dir) -> None:
        model_files = glob.glob(
            f"{checkpoint_dir}/models/*"
        )  # * means all if need specific format then *.csv
        latest_file = max(model_files, key=os.path.getctime)

        OmegaConf.set_struct(cfg, False)
        cfg.checkpoint = latest_file
        OmegaConf.set_struct(cfg, True)


    @abstractmethod
    def chat(self, messages: ChatHistoryType | List[ChatHistoryType]) -> List[str]:
        pass

    @abstractmethod
    def chat_stream(self, messages: ChatHistoryType) -> Iterator[str]:
        pass
