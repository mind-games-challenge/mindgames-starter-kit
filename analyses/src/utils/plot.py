import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def heatmap(data, row_labels, col_labels, step: int = 5, rotation: int = 90, vmin: float=None, vmax: float = None, ax=None, cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data
            A 2D numpy array of shape (M, N).
        row_labels
            A list or array of length M with the labels for the rows.
        col_labels
            A list or array of length N with the labels for the columns.
        ax
            A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
            not provided, use current Axes or create a new one.  Optional.
        cbar_kw
            A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        cbarlabel
            The label for the colorbar.  Optional.
        **kwargs
            All other arguments are forwarded to `imshow`.

    Notes:
        retrieved from: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    if vmin and vmax:
        im.set_clim(vmin, vmax)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation="vertical", va="top")

    # Show all ticks and label them with the respective list entries.
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xticks(
        np.arange(0, data.shape[1], step),
        labels=col_labels,
        rotation=rotation,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticks(np.arange(0, data.shape[0], step), labels=row_labels)

    # Let the horizontal axes labeling appear on bottom.
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(0, data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(0, data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def add_patches(offsets: np.array, ax: plt.Axes = None):
    """
    Annotate the ``ax`` by adding square patches at each offset.

    Args:
        offsets (np.array): an array of patch offsets.
        ax (plt.Axes): an ax object.

    Returns:

    """
    if ax is None:
        ax = plt.gca()

    patch_x = -0.5
    patch_y = -0.5

    for value in offsets:
        ax.add_patch(
            patches.Rectangle(
                (patch_x, patch_y),
                value,
                value,
                edgecolor="red",
                fill=False,
                lw=2,
                zorder=3,
            )
        )
        patch_y = patch_y + value
        patch_x = patch_x + value

    red_patch = patches.Patch(edgecolor="red", fill=False, lw=1, label="Game boundary")
    ax.legend(handles=[red_patch])

    return ax
