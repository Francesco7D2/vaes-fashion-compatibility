# -*- coding: utf-8 -*-


import yaml
import logging



def load_config(config_path='config.yml'):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config


def setup_logging(log_file='reports/logs/main.log', log_level=logging.INFO):
    logging.basicConfig(filename=log_file, level=log_level,
                        format='%(asctime)s - %(levelname)s - %(message)s')
                        
