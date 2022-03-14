from omegaconf import OmegaConf
import os
import re
def parse_cfg(cfg_path: str = None, default_cfg_path: str = None) -> OmegaConf:
    """Parses a config file and returns an OmegaConf object."""
    if default_cfg_path is not None:
        base = OmegaConf.load(default_cfg_path)
    else:
        base = OmegaConf.create()
    cli = OmegaConf.from_cli()
    for k, v in cli.items():
        if v is None:
            cli[k] = True
    base.merge_with(cli)
   
    if cfg_path is not None:
        cfg = OmegaConf.load(cfg_path)
        base.merge_with(cfg)

    return base