import torch.nn as nn

# Deep learning methods
from src.modelzoo.cudalstm import CudaLSTM
from src.utils.config import Config


def get_model(cfg: Config) -> nn.Module:
    """Get model object, depending on the run configuration.

    Parameters
    ----------
    cfg : Config
        The run configuration.

    Returns
    -------
    nn.Module
        A new model instance of the type specified in the config.
    """

    if cfg.model.lower() == "cudalstm":
        model = CudaLSTM(cfg=cfg)
    else:
        raise NotImplementedError(f"{cfg.model} not implemented or not linked in `get_model()`")

    return model
