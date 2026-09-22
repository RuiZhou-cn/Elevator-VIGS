import yaml
import rich
import copy
import torch
from config_defaults import DEFAULTS

_log_styles = {
    "GSBackend": "bold green",
    "GUI": "bold magenta",
    "Eval": "bold red",
    "PGBA": "bold blue",
}


def get_style(tag):
    if tag in _log_styles.keys():
        return _log_styles[tag]
    return "bold blue"


def Log(*args, tag="GSBackend"):
    style = get_style(tag)
    rich.print(f"[{style}]{tag}:[/{style}]", *args)


def load_config(path):
    """ Loads config file: the dataset's yaml merged over DEFAULTS (config_defaults.py). """
    with open(path, "r") as f:
        cfg_special = yaml.full_load(f)
    cfg = copy.deepcopy(DEFAULTS)
    update_recursive(cfg, cfg_special)
    return cfg


def update_recursive(dict1, dict2):
    """ Update two config dictionaries recursively. dict1 get masked by dict2, and we return dict1. """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def clone_obj(obj):
    clone_obj = copy.deepcopy(obj)
    for attr in clone_obj.__dict__.keys():
        # check if its a property
        if hasattr(clone_obj.__class__, attr) and isinstance(
            getattr(clone_obj.__class__, attr), property
        ):
            continue
        if isinstance(getattr(clone_obj, attr), torch.Tensor):
            setattr(clone_obj, attr, getattr(clone_obj, attr).detach().clone())
    return clone_obj
