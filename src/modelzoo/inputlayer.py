from typing import Optional

import torch
import torch.nn as nn

from src.utils.config import Config


class InputLayer(nn.Module):
    """Input layer to preprocess static and dynamic inputs.

    This layer prepares the data before passing it to the main models. This can include running the dynamic and static
    attributes through embedding networks, preprocessing and assembling data at different temporal frequencies (e.g.
    daily, hourly), doing probabilistic masking and handling missing data.

    In the simplest case, the layer takes the dictionary containing the sample information and assembles the tensor to
    be sent to the main model.

    Parameters
    ----------
    cfg : Config
        Configuration file.

    """

    def __init__(self, cfg: Config):
        super().__init__()

        self.dynamic_input = cfg.dynamic_input
        self._x_d_key = "x_d"

        # Get dynamic input size
        self.dynamic_input_size = (
            len(self.dynamic_input) if cfg.dynamic_embedding is None else cfg.dynamic_embedding["hiddens"][-1]
        )

        # Get static input size
        if not cfg.static_input:
            self.static_input_size = 0
        elif isinstance(cfg.static_input, list) and cfg.static_embedding is None:
            self.static_input_size = len(cfg.static_input)
        else:
            self.static_input_size = cfg.static_embedding["hiddens"][-1]

        # Get embedding networks
        self._get_embeddings(cfg)

        # Output size of the input layer
        self.output_size = self.dynamic_input_size + self.static_input_size

        # Save config
        self.cfg = cfg

    def forward(
        self, sample: dict[str, torch.Tensor | dict[str, torch.Tensor]], assemble: bool = True
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Forward pass of embedding networks.

        Parameters
        ----------
        sample: dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Dictionary with the different tensors / dictionaries that will be used for the forward pass.

        assemble: bool
            Whether to assemble the different tensors into a single tensor or return a dictionary with the different

        Returns
        -------
        torch.Tensor | dict[str, torch.Tensor]
            Either the processed tensor or a dictionary with the different tensors that then have the be assembled
            manually

        """
        # -------------------------
        # Dynamic inputs
        # -------------------------
        x_d = torch.cat([v(torch.stack(list(sample[k].values()), dim=-1)) for k, v in self.emb_x_d.items()], dim=1)


        # -------------------------
        # Static inputs
        # -------------------------
        x_s = (
            self.emb_x_s(sample["x_s"]).unsqueeze(1).expand(-1, x_d.shape[1], -1)
            if self.cfg.static_input
            else x_d.new_zeros(x_d.shape[0], x_d.shape[1], 0)
        )

        return torch.cat([x_d, x_s], dim=2) if assemble else {"x_d": x_d, "x_s": x_s}

    def _get_embeddings(self, cfg: Config):
        """Build embedding networks based on the configuration.

        Parameters
        ----------
        cfg : Config
            Configuration file.

        """

        # -------------------------
        # Embeddings for dynamic variables
        # -------------------------
        self.emb_x_d = nn.ModuleDict()

        # Case 1: Single group of variables, same variables along the sequence length, and only one sequence processing.
        # This is the only case existing in student version of hy2dl
        if isinstance(self.dynamic_input, list):
            self.emb_x_d[self._x_d_key] = InputLayer.build_embedding(
                input_dim=len(self.dynamic_input), embedding=cfg.dynamic_embedding
            )
        else:
            raise ValueError("dynamic_input must be a list in hindcast-only mode.")

        # -------------------------
        # Embeddings for static variables
        # -------------------------
        if cfg.static_input:
            self.emb_x_s = InputLayer.build_embedding(input_dim=len(cfg.static_input), embedding=cfg.static_embedding)

    @staticmethod
    def build_embedding(input_dim: int, embedding: Optional[dict[str, str | float | list[int]]]):
        """Build embedding

        Parameters
        ----------
        input_dim: int
            Input dimension of the first layer.
        embedding: dict[str, str | float | list[int]]
            Dictionary with the embedding characteristics

        Returns
        -------
        nn.Sequential | nn.Identity
            Embedding network or nn.Identity

        """

        return (
            InputLayer.build_ffnn(
                input_dim=input_dim,
                spec=embedding["hiddens"],
                activation=embedding["activation"],
                dropout=embedding["dropout"],
            )
            if isinstance(embedding, dict)
            else nn.Identity()
        )

    @staticmethod
    def build_ffnn(input_dim: int, spec: list[int], activation: str = "relu", dropout: float = 0.0) -> nn.Sequential:
        """Builds a feedforward neural network based on the given specification.

        Parameters
        ----------
        input_dim: int
            Input dimension of the first layer.
        spec: List[int]
            Dimension of the different hidden layers.
        activation: str
            Activation function to use between layers (relu, linear, tanh, sigmoid).
            Default is 'relu'.
        dropout: float
            Dropout rate to apply after each layer (except the last one).
            Default is 0.0 (no dropout).

        Returns
        -------
        nn.Sequential
            A sequential model containing the feedforward neural network layers.

        """

        activation = InputLayer._get_activation_function(activation)
        ffnn_layers = []
        for i, out_dim in enumerate(spec):
            ffnn_layers.append(nn.Linear(input_dim, out_dim))
            if i != len(spec) - 1:  # add activation, except after the last linear
                ffnn_layers.append(activation)
                if dropout > 0.0:
                    ffnn_layers.append(nn.Dropout(dropout))

            input_dim = out_dim  # updates next layer’s input size

        return nn.Sequential(*ffnn_layers)

    @staticmethod
    def _get_activation_function(activation: str) -> nn.Module:
        """Returns the activation function based on the given string.

        Parameters
        ----------
        activation: str
            Name of the activation function (e.g., 'relu', 'linear', 'tanh', 'sigmoid').

        Returns
        -------
        nn.Module
            The corresponding activation function module.

        """

        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation == "linear":
            return nn.Identity()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
