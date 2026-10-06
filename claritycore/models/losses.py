# Copyright (c) Aman Urumbekov and other contributors.
"""Loss functions for ClarityCore models."""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

Reduction = Literal["none", "mean", "sum"]


class L1Loss(nn.Module):
    """L1 (Mean Absolute Error) loss."""

    def __init__(self, reduction: Reduction = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target, reduction=self.reduction)


class MSELoss(nn.Module):
    """MSE (Mean Squared Error) loss."""

    def __init__(self, reduction: Reduction = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target, reduction=self.reduction)


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss (differentiable L1 variant).

    L = sqrt((pred - target)² + ε)

    More robust than L1 for small errors.

    Args:
        eps: Small constant for numerical stability.
        reduction: Reduction method.
    """

    def __init__(self, eps: float = 1e-6, reduction: Reduction = "mean") -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class SmoothL1Loss(nn.Module):
    """
    Smooth L1 (Huber) loss.

    Combines L1 and L2 loss - L2 for small errors, L1 for large.

    Args:
        beta: Threshold for switching from L2 to L1.
        reduction: Reduction method.
    """

    def __init__(self, beta: float = 1.0, reduction: Reduction = "mean") -> None:
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(pred, target, beta=self.beta, reduction=self.reduction)


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features.

    Compares high-level features from a pretrained VGG network.

    Args:
        layer_weights: Dict mapping layer name to weight.
        use_input_norm: Normalize inputs to ImageNet stats.
        reduction: Reduction method.
    """

    def __init__(
        self,
        layer_weights: dict[str, float] | None = None,
        use_input_norm: bool = True,
        reduction: Reduction = "mean",
    ) -> None:
        super().__init__()

        self.layer_weights = layer_weights or {
            "conv1_2": 0.1,
            "conv2_2": 0.1,
            "conv3_4": 1.0,
            "conv4_4": 1.0,
            "conv5_4": 1.0,
        }
        self.use_input_norm = use_input_norm
        self.reduction = reduction

        # Load VGG lazily to avoid import overhead
        self._vgg = None

        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _load_vgg(self, device: torch.device) -> nn.Module:
        """Lazily load VGG model."""
        if self._vgg is None:
            from torchvision.models import VGG19_Weights, vgg19

            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
            vgg.eval()
            for p in vgg.parameters():
                p.requires_grad_(False)

            self._vgg = vgg.to(device)

        return self._vgg

    def _extract_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract VGG features at specified layers."""
        vgg = self._load_vgg(x.device)

        if self.use_input_norm:
            x = (x - self.mean) / self.std

        features = {}
        layer_name_mapping = {
            3: "conv1_2",
            8: "conv2_2",
            17: "conv3_4",
            26: "conv4_4",
            35: "conv5_4",
        }

        for i, layer in enumerate(vgg):
            x = layer(x)
            if i in layer_name_mapping:
                name = layer_name_mapping[i]
                if name in self.layer_weights:
                    features[name] = x

        return features

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_features = self._extract_features(pred)
        target_features = self._extract_features(target)

        loss = torch.tensor(0.0, device=pred.device)

        for name, weight in self.layer_weights.items():
            if name in pred_features:
                layer_loss = F.l1_loss(
                    pred_features[name],
                    target_features[name],
                    reduction=self.reduction,
                )
                loss = loss + weight * layer_loss

        return loss


class GANLoss(nn.Module):
    """
    GAN loss for discriminator training.

    Supports multiple GAN variants.

    Args:
        gan_type: Type of GAN loss ('vanilla', 'lsgan', 'wgan', 'hinge').
        real_label: Target value for real samples.
        fake_label: Target value for fake samples.
    """

    def __init__(
        self,
        gan_type: Literal["vanilla", "lsgan", "wgan", "hinge"] = "vanilla",
        real_label: float = 1.0,
        fake_label: float = 0.0,
    ) -> None:
        super().__init__()

        self.gan_type = gan_type
        self.real_label = real_label
        self.fake_label = fake_label

        if gan_type == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_type == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_type in ("wgan", "hinge"):
            self.loss = None
        else:
            raise ValueError(f"Unknown GAN type: {gan_type}")

    def forward(
        self,
        pred: torch.Tensor,
        target_is_real: bool,
        is_discriminator: bool = False,
    ) -> torch.Tensor:
        """
        Compute GAN loss.

        Args:
            pred: Discriminator output.
            target_is_real: Whether target is real or fake.
            is_discriminator: Whether computing for discriminator or generator.
        """
        if self.gan_type in ("vanilla", "lsgan"):
            target = pred.new_full(pred.shape, self.real_label if target_is_real else self.fake_label)
            return self.loss(pred, target)

        elif self.gan_type == "wgan":
            return -pred.mean() if target_is_real else pred.mean()

        elif self.gan_type == "hinge":
            if is_discriminator:
                if target_is_real:
                    return F.relu(1 - pred).mean()
                else:
                    return F.relu(1 + pred).mean()
            else:
                return -pred.mean()


class FocalFrequencyLoss(nn.Module):
    """
    Focal Frequency Loss (Jiang et al., ICCV 2021).

    Computes weighted L2 distance in the frequency domain. The weight is proportional to the magnitude of the error itself, so that the loss focuses on frequencies where the model is currently failing.

    Args:
        alpha: Focusing parameter. Higher alpha = more focus on hard frequencies. Default 1.0 (recommended in the paper).
        loss_weight: Overall scaling of the loss.
        patch_factor: Use patch-based FFT for larger images. 1 = global FFT.
        ave_spectrum: If True, use magnitude-only (spectrum) instead of complex coefficients. Default False (keeps phase).
        log_matrix: Apply log1p to the amplitude weights. Default False.
        batch_matrix: Normalize the weights per batch. Default False.
        reduction: 'mean' or 'sum'.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        loss_weight: float = 1.0,
        patch_factor: int = 1,
        ave_spectrum: bool = False,
        log_matrix: bool = False,
        batch_matrix: bool = False,
        reduction: Reduction = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.loss_weight = loss_weight
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix
        self.reduction = reduction

    def _fft2d(self, x: torch.Tensor) -> torch.Tensor:
        """2D FFT, shifted to center low frequencies at the middle."""
        return torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"))

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Split spatial dims into a grid of patches for local FFT."""
        p = self.patch_factor

        if p == 1:
            return x
        
        b, c, h, w = x.shape
        assert h % p == 0 and w % p == 0, \
            f"spatial dims ({h},{w}) must be divisible by patch_factor={p}"
        
        x = x.view(b, c, p, h // p, p, w // p)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        return x.view(b, c * p * p, h // p, w // p)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_p = self._patchify(pred)
        target_p = self._patchify(target)

        f_pred = self._fft2d(pred_p)
        f_target = self._fft2d(target_p)

        diff = f_pred - f_target

        weight = torch.abs(diff).pow(self.alpha)
        weight = weight / (weight.mean(dim=(1, 2, 3), keepdim=True) + 1e-8)

        if self.log_matrix:
            weight = torch.log1p(weight)

        if self.batch_matrix:
            wmin = weight.amin(dim=(0, 2, 3), keepdim=True)
            wmax = weight.amax(dim=(0, 2, 3), keepdim=True)
            weight = (weight - wmin) / (wmax - wmin + 1e-8)

        loss_map = weight * (diff.real ** 2 + diff.imag ** 2)

        if self.reduction == "mean":
            loss = loss_map.mean()
        elif self.reduction == "sum":
            loss = loss_map.sum()
        else:
            loss = loss_map

        return self.loss_weight * loss


def get_loss(name: str, **kwargs) -> nn.Module:
    """
    Get a loss function by name.

    Args:
        name: Loss name ('l1', 'mse', 'charbonnier', 'perceptual', 'gan').
        **kwargs: Loss-specific arguments.

    Returns:
        Loss module.
    """
    losses = {
        "l1": L1Loss,
        "mse": MSELoss,
        "charbonnier": CharbonnierLoss,
        "smooth_l1": SmoothL1Loss,
        "perceptual": PerceptualLoss,
        "gan": GANLoss,
        "ffl": FocalFrequencyLoss,
    }

    if name not in losses:
        raise ValueError(f"Unknown loss: {name}. Available: {list(losses.keys())}")

    return losses[name](**kwargs)


__all__ = [
    "L1Loss",
    "MSELoss",
    "CharbonnierLoss",
    "SmoothL1Loss",
    "PerceptualLoss",
    "GANLoss",
    "FocalFrequencyLoss",
    "get_loss",
]
