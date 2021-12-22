from abc import ABC

import torch
import abc


class InputPerturbation(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def __str__(self) -> str:
        """
        Returns: Name of the associated pretext task
        """

    @abc.abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the perturbed
        Parameters
        ----------
        x : torch.Tensor
            Input data
        """


class Identity(InputPerturbation, ABC):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Reconstruction"

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x


class RandomNoise(InputPerturbation, ABC):
    def __init__(self, noise_level: float):
        super().__init__()
        self.noise_level = noise_level

    def __str__(self):
        return "Denoising"

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.noise_level*torch.randn(x.shape, device=x.device)


class Mask(InputPerturbation, ABC):
    def __init__(self, mask_proportion: float = 0.2, baseline: float = 0):
        super().__init__()
        self.baseline = baseline
        self.mask_proportion = mask_proportion

    def __str__(self):
        return "Inpainting"

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ber_tensor = torch.bernoulli((1-self.mask_proportion)*torch.ones(x.shape, device=x.device))
        x = ber_tensor*x + (1-ber_tensor)*self.baseline*torch.ones(x.shape, device=x.device)
        return x




