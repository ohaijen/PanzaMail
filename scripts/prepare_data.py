import copy
from datetime import datetime
import json
import logging
import os
import random
import shutil
import time
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from panza import PanzaWriter  # The import also loads custom Hydra resolvers
from panza.data_preparation.data_preparation import load_documents, generate_synthetic_instructions, check_if_file_exists, split_and_write_data

LOGGER = logging.getLogger(__name__)


def rename_config_keys(cfg: DictConfig) -> None:
    # Disable struct mode to allow modifications
    OmegaConf.set_struct(cfg, False)

    cfg.writer.llm.sampling_parameters = cfg.writer.llm.sampling
    del cfg.writer.llm.sampling

    cfg.writer.prompt_builder = cfg.writer.prompting
    del cfg.writer.prompting

    # Re-enable struct mode to lock down the configuration
    OmegaConf.set_struct(cfg, True)




@hydra.main(version_base="1.1", config_path="../configs", config_name="panza_preparation")
def main(cfg: DictConfig) -> None:
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



if __name__ == "__main__":
    main()
