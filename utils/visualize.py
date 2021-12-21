from captum.attr._utils.visualization import visualize_image_attr
import torch
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_image_saliency(image: torch.Tensor, saliency: torch.Tensor):
    image_np = image.permute((1, 2, 0)).cpu().numpy()
    saliency_np = saliency.permute((1, 2, 0)).cpu().numpy()
    visualize_image_attr(saliency_np, image_np)


def plot_vae_saliencies(images: list, saliency: np.ndarray) -> plt.Figure:
    W = saliency.shape[-1]
    n_plots = len(saliency)
    dim_latent = saliency.shape[1]
    cblind_palette = sns.color_palette("colorblind")
    fig, axs = plt.subplots(ncols=dim_latent+1, nrows=n_plots, figsize=(3 * (dim_latent+1), 3 * n_plots))
    for example_id in range(n_plots):
        max_saliency = np.max(saliency[example_id])
        ax = axs[example_id, 0]
        ax.imshow(images[example_id], cmap='gray')
        ax.axis('off')
        ax.set_title('Original Image')
        for dim in range(dim_latent):
            sub_saliency = saliency[example_id, dim]
            ax = axs[example_id, dim+1]
            h = sns.heatmap(np.reshape(sub_saliency, (W, W)), linewidth=0, xticklabels=False, yticklabels=False,
                            ax=ax, cmap=sns.light_palette(cblind_palette[dim], as_cmap=True), cbar=False,
                            alpha=1, zorder=2, vmin=0, vmax=max_saliency)
            ax.set_title(f'Saliency Dimension {dim+1}')
    return fig


def vae_box_plots(df: pd.DataFrame, metric_names: list) -> plt.Figure:
    fig, axs = plt.subplots(ncols=1, nrows=len(metric_names), figsize=(6, 4 * len(metric_names)))
    for id_metric, metric in enumerate(metric_names):
        sns.boxplot(data=df, x="Beta", y=metric, hue="Loss Type", palette="colorblind", ax=axs[id_metric])
    return fig
