'''
Contains helper functions used throughout the project.
'''
import os
import yaml
from pathlib import Path

def get_config() -> dict:
    '''
    For loading .yaml configuration file.
    Returns: dictionary of configuration variables.
    '''
    config_path = Path.cwd()/'config.yml'
    config_path = Path(os.path.dirname(__file__))/'config.yml'
    config = yaml.full_load(config_path.read_bytes())
    return config