import logging

import glob
import hydra
import os
from omegaconf import DictConfig, OmegaConf

from panza import PanzaWriter  # The import also loads custom Hydra resolvers

LOGGER = logging.getLogger(__name__)

def set_latest_model(cfg: DictConfig) -> None:
    model_files = glob.glob(
        f"{cfg.checkpoint_dir}/models/*"
    )  # * means all if need specific format then *.csv
    latest_file = max(model_files, key=os.path.getctime)

    OmegaConf.set_struct(cfg, False)
    cfg.interfaces.writer.llm.checkpoint = latest_file
    OmegaConf.set_struct(cfg, True)


@hydra.main(version_base="1.1", config_path="../configs", config_name="panza_writer")
def main(cfg: DictConfig) -> None:
    LOGGER.info("Starting Panza Writer")
    

    # Find the latest checkpoint, if requested.
    if cfg.interfaces.writer.llm.checkpoint == "latest":
        set_latest_model(cfg)

    print(cfg.interfaces)

    hydra.utils.instantiate(cfg.interfaces)

if __name__ == "__main__":
    main()
