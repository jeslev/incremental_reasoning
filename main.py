import random, os
import argparse

from wrapper import Config
from Winogrande import Winogrande

import logger

log = logger.setup_applevel_logger(file_name = 'winogrande.log')


config_file = "config/default.yaml"
random.seed(42)

def main(cfg):

    # implement winogrande algorithm
    config_winogrande = {
        "ensemble_size": cfg.get("setting/winogrande/n_classifiers"),
        "training_set_size": cfg.get("setting/winogrande/training_set_size"),
        "cutoff_size": cfg.get("setting/winogrande/top-k"),
        "threshold": cfg.get("setting/winogrande/threshold")
    }

    winogrande = Winogrande(cfg) # generate/load embeddings
    log.debug("Applying Winogrande algorithm")
    winogrande.run(**config_winogrande)

    log.debug("New dataset has " + str(len(winogrande._filtered_dataset)) + " entries.")

    output_path = os.path.join("./output", cfg.name)
    winogrande.save(output_path)


def print_params(cfg):
    log.debug("Config file configuration")
    log.debug(cfg.__str__())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Choose the config file.")
    args = parser.parse_args()

    if args.config:
        cfg = Config(args.config, default_path=config_file)
    else:
        cfg = Config(config_file)

    print_params(cfg)
    main(cfg)

