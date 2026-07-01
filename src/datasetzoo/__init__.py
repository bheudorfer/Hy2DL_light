from src.datasetzoo.basedataset import BaseDataset
from src.datasetzoo.camelsde import CAMELS_DE
from src.utils.config import Config


def get_dataset(cfg: Config) -> BaseDataset:
    """Get data set instance, depending on the run configuration.

    This class and its methods are based on Neural Hydrology [#]_ and adapted for our specific case.

    Parameters
    ----------
    cfg : Config
        Configuration file.

    References
    ----------
    .. [#] F. Kratzert, M. Gauch, G. Nearing and D. Klotz: NeuralHydrology -- A Python library for Deep Learning
        research in hydrology. Journal of Open Source Software, 7, 4050, doi: 10.21105/joss.04050, 2022
    """
    if cfg.dataset.lower() == "camels_de":
        Dataset = CAMELS_DE
    else:
        raise NotImplementedError(f"No dataset class implemented for dataset {cfg.dataset}")

    return Dataset
