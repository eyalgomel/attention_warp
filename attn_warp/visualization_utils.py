from pathlib import Path
import cv2
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import gridspec
import seaborn as sns


def visualize_attention_output(attention_output, method='mean', num_heads=8, h=64, w=64, name='', cmap='RdBu', save_path=None):
    """
    Visualizes the cross-attention output.

    Parameters:
    - attention_output: np.ndarray
        A numpy array of shape (pixels, features_dim), representing the cross-attention output.
    - method: str, optional
        How to aggregate across heads and dimensions. Options are 'mean', 'sum', or 'head-wise'.
    - num_heads: int, optional
        Number of attention heads (default is 8).

    Returns:
    - None: Displays the attention maps.
    """

    if save_path is not None:
        action = lambda: None
    else:
        action = lambda: plt.show()


    # Reshape the pixels into a spatial grid, h x w x num_heads x head_dim
    spatial_grid = attention_output.reshape(h, w, num_heads, -1)

    if method == 'mean':

        aggregated_map = np.mean(spatial_grid, axis=(2, 3))  # Mean across heads and dimensions
        plt.axis('off')
        plt.tight_layout()
        sns.heatmap(aggregated_map, cmap=cmap, square=True, cbar=False,).get_figure().savefig(save_path,
                                                                                              bbox_inches='tight',pad_inches=0)
        plt.close()

    elif method == 'sum':
        # Sum across heads and dimensions
        aggregated_map = np.sum(spatial_grid, axis=(2, 3))  # Sum across heads and dimensions
        plt.figure(figsize=(6, 6))
        #cmap blue and red cmap='RdBu
        sns.heatmap(aggregated_map, cmap=cmap, square=True, cbar=True)
        plt.title(f'{name} | Sum | Res {h}x{w} | Channels {spatial_grid.shape[3]}')
        plt.axis('off')
        action()

    elif method == 'head-wise':
        # Visualize each head separately
        fig, axes = plt.subplots(1, num_heads, figsize=(16, 4))
        for head in range(num_heads):
            head_map = np.mean(spatial_grid[:, :, head, :], axis=2)  # Mean across head_dim
            sns.heatmap(head_map, ax=axes[head], cmap=cmap, square=True, cbar=False)
            axes[head].set_title(f'Head {head+1}')
            axes[head].axis('off')
        plt.tight_layout()
        action()

    elif method == 'combined':
        # Combined plot: Mean map + individual head maps in the same figure
        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(2, num_heads, height_ratios=[2, 1], width_ratios=[1]*num_heads)

        # Mean map calculation
        aggregated_map = np.mean(spatial_grid, axis=(2, 3))  # Mean across heads and dimensions

        # Create the mean plot across multiple axes (first row, spanning all columns)
        middle_start = num_heads // 4
        middle_end = num_heads - middle_start
        ax_mean = plt.subplot(gs[0, middle_start:middle_end])
        sns.heatmap(aggregated_map, cmap=cmap, square=True, cbar=True, ax=ax_mean, cbar_kws={'label': 'Mean Attention'})
        ax_mean.set_title(f'{name} | Mean | Res {h}x{w} | Channels {spatial_grid.shape[3]}')
        ax_mean.axis('off')

        # Plot each head's map in the second row
        for head in range(num_heads):
            ax = plt.subplot(gs[1, head])
            head_map = np.mean(spatial_grid[:, :, head, :], axis=2)  # Mean across head_dim
            sns.heatmap(head_map, cmap=cmap, square=True, cbar=False, ax=ax)
            ax.set_title(f'Head {head+1}', fontsize=12)
            ax.axis('off')

        plt.tight_layout()
        action()

        plt.close()


def create_videos_for_maps(images_folder="", fps=10):
    # Create output folder for videos
    video_dir = Path(images_folder) / 'videos'
    video_dir.mkdir(exist_ok=True)  # Create directory if it doesn't exist
    
    # Get all maps that match the pattern
    images = sorted(images_folder.glob("map_*.png"))  # List of images

    # Group images by map_idx1
    grouped_images = {}
    for image_file in images:
        idx1 = image_file.stem.split('_')[1]  # Extract idx1 from the filename
        if idx1 not in grouped_images:
            grouped_images[idx1] = []
        grouped_images[idx1].append(image_file)

    img_example = cv2.imread(str(images[0]))
    height, width, _ = img_example.shape
    # Create a video for each unique map_idx1
    for idx1, files in grouped_images.items():
        video_name = video_dir / f'video_map_{idx1}.mp4'  # Video filename
        video_writer = cv2.VideoWriter(str(video_name), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        # Write each image to the video
        for image_file in sorted(files)[::-1]:
            img = cv2.imread(str(image_file))
            video_writer.write(img)

        video_writer.release()
