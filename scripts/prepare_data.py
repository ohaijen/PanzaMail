import copy
from datetime import datetime
import json
import logging
import os
import random
import shutil
import time
from pathlib import Path
from typing import List, Optional, Sequence

import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from panza import PanzaWriter  # The import also loads custom Hydra resolvers

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
CONFIG_NAME = "panza_preparation"


def rename_config_keys(cfg: DictConfig) -> None:
    # Disable struct mode to allow modifications
    OmegaConf.set_struct(cfg, False)

    cfg.writer.llm.sampling_parameters = cfg.writer.llm.sampling
    del cfg.writer.llm.sampling

    cfg.writer.prompt_builder = cfg.writer.prompting
    del cfg.writer.prompting

    # Re-enable struct mode to lock down the configuration
    OmegaConf.set_struct(cfg, True)




def compose_config(overrides: Optional[Sequence[str]] = None) -> DictConfig:
    overrides = list(overrides or [])
    if not any(override.lstrip("+").startswith("panza_workspace=") for override in overrides):
        overrides.insert(0, f"panza_workspace={REPO_ROOT}")

    if GlobalHydra.instance().is_initialized():
        cfg = hydra.compose(
            config_name=CONFIG_NAME,
            overrides=overrides,
            return_hydra_config=True,
        )
    else:
        with hydra.initialize_config_dir(version_base="1.1", config_dir=str(CONFIG_DIR)):
            cfg = hydra.compose(
                config_name=CONFIG_NAME,
                overrides=overrides,
                return_hydra_config=True,
            )

    HydraConfig.instance().set_config(cfg)
    OmegaConf.set_struct(cfg, False)
    del cfg["hydra"]
    OmegaConf.set_struct(cfg, True)
    return cfg


def main(cfg: Optional[DictConfig] = None, overrides: Optional[Sequence[str]] = None) -> None:
    if cfg is None:
        cfg = compose_config(overrides)

    print("*"*20 + "\n CREATING TRAINING DATA WITH CONFIG:")
    print(cfg)
    print("*"*20)

    from panza.data_preparation.data_preparation import (
        generate_synthetic_instructions,
        load_documents,
        split_and_write_data,
    )

    LOGGER.info("Running Panza Data Preparation")
    LOGGER.info("Configuration: \n%s", OmegaConf.to_yaml(cfg, resolve=True))

    # Rename config keys to follow class structure
    #rename_config_keys(cfg)

    # # Skip running if  already exist
    # if not check_if_file_exists(cfg):
    #     # Extract the emails from the .mbox file
    #     extract_snippets(
    #         cfg.email_dump_path,
    #         cfg.cleaned_emails_path,
    #         #[cfg.user.email_address],
    #         cfg.discarded_emails_dir,
    #     )


    writer: PanzaWriter = hydra.utils.instantiate(cfg.writer)
    assert isinstance(writer, PanzaWriter), "Failed to instantiate PanzaWriter"



    madlibs = [
        ["child", {"style": "childlike",
        "description": "naive and simple, with no long words and some irrelevant info",
        "filename": '{filename}',
        "snippet": '{snippet}',
        }],
        ["professional", {"style": "polite, professional, formal, textbook",
         "description": "impersonal, like a textbook example of how such a snippet might sound",
        "filename": '{filename}',
        "snippet": '{snippet}',
         }]
    ]
    


    # Load documents
    documents = load_documents(cfg.cleaned_emails_path)
    num_cycles = 2
    for cycle_num in range(num_cycles):
        for document in documents:
            document.snippet_text = document.original_text
        if madlibs != {}:
            import copy
            default_prompt = copy.deepcopy(writer.prompt_builder.summarization_prompt)

            # The first madlib goes from the loaded documents to a file that looks like the final output, but is underwritten also with the name
            # The second one goes from the snippets in that first output to the a file that looks like the final output, but...
            # The last one also needs to be written to a file with no _name at the end.
            num_iterations = len(madlibs)
            for i, [name, madlib] in enumerate(madlibs):
                print(default_prompt, madlib)
                writer.prompt_builder.summarization_prompt = default_prompt.format(**madlib)

                if i < num_iterations - 1:
                    output_path = cfg.summarized_emails_path.replace(".jsonl", f"_{name}.jsonl")
                else:
                    output_path = cfg.summarized_emails_path

                output_path = output_path.replace(".jsonl", f"_cycle{cycle_num}.jsonl")

                documents = generate_synthetic_instructions(
                    documents=documents,
                    writer=writer,
                    batch_size=cfg.batch_size,
                    output_path=output_path,
                    update_documents = True
                )
        else:
            output_path = cfg.summarized_emails_path.replace(".jsonl", f"_cycle{cycle_num}.jsonl")

            generate_synthetic_instructions(
                documents=documents,
                writer=writer,
                batch_size=cfg.batch_size,
                output_path=cfg.summarized_emails_path,
            )

    # Write the test data to test.jsonl, with an optional train-test split
    split_and_write_data(cfg)



@hydra.main(version_base="1.1", config_path="../configs", config_name=CONFIG_NAME)
def hydra_main(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_main()
