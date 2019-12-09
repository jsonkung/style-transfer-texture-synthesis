import argparse
import cv2
import os
import numpy as np

import pca_nn
from pyramid import get_pyramid
from irls import aggregate_patches
from color_transfer import color_transfer
from segment import get_segmentation_mask
from fuse_content import fuse_content
from skimage.restoration import denoise_tv_chambolle




####### - begin github additions

from pca import pca, project
from sklearn.feature_extraction.image import extract_patches
from sklearn.neighbors import NearestNeighbors

LMAX = 3
IM_SIZE = 400
PATCH_SIZES = np.array([33, 21, 13, 9])
SAMPLING_GAPS = np.array([28, 18, 8, 5])
IALG = 10
IRLS_it = 3
IRLS_r = 0.8
PADDING_MODE = 'edge'
# from github: implemented color transfer with histogram matching using interpolation
def color_transfer(content, style):
    transfered = np.copy(content)
    # for each channel of the content, match the cum_histogram with the style's one
    for i in range(0, content.shape[2]):
        content_channel = content[:, :, i].flatten()
        style_channel = style[:, :, i].flatten()
        # calculate histogram for both content and style
        content_values, content_indices, content_counts = np.unique(content_channel, return_inverse=True, return_counts=True)
        style_values, style_counts = np.unique(style_channel, return_counts=True)
        # calculate cummulative histogram
        content_cumhist = np.cumsum(content_counts)
        style_cumhist = np.cumsum(style_counts)
        # normalize it
        content_cumhist = content_cumhist / np.max(content_cumhist)
        style_cumhist = style_cumhist / np.max(style_cumhist)
        # match using interpolation
        matched = np.interp(content_cumhist, style_cumhist, style_values)
        transfered[:, :, i] = matched[content_indices].reshape(content[:, :, i].shape)
    return transfered

def solve_irls(X, X_patches_raw, patch_size, sampling_gap, style_patches, neighbors, projection_matrix):
    # current_size = X.shape[0]
    # Extracting Patches
    X_patches = X_patches_raw.reshape(-1, patch_size * patch_size * 3)
    npatches = X_patches.shape[0]
    if patch_size <= 21:
        X_patches = project(X_patches, projection_matrix)  # Projecting X to same dimention as style patches
    # Computing Nearest Neighbors
    distances, indices = neighbors.kneighbors(X_patches)
    distances += 0.0001
    # Computing Weights
    weights = np.power(distances, IRLS_r - 2)
    # Patch Accumulation
    R = np.zeros((X.shape[0], X.shape[1], 3), dtype=np.float32)
    Rp = extract_patches(R, patch_shape=(patch_size, patch_size, 3), extraction_step=sampling_gap)
    print("Rp shape:", Rp.shape)
    X[:] = 0
    t = 0
    for t1 in range(X_patches_raw.shape[0]):
        for t2 in range(X_patches_raw.shape[1]):
            nearest_neighbor = style_patches[indices[t, 0]]
            X_patches_raw[t1, t2, 0, :, :, :] += nearest_neighbor * weights[t]
            Rp[t1, t2, 0, :, :, :] += 1 * weights[t]
            t = t + 1
    R += 0.0001  # to avoid dividing by zero.
    X /= R

######## - end github additions

def style_transfer(content, style, num_pyramid_layers=3, patch_sizes=(33, 21, 13, 9, 5), patch_spacings=(28, 18, 8, 5, 3), num_iters=3, num_irls_iters=10, max_pixel_val=255):
    """
    Perform style transfer via texture synthesis with the following steps
    1. Start at lowest resolution layer of pyramid
    2. Iterate from largest to smallest patch size
    3. Perform nearest neighbors to find patches in style with smallest L2 norm
    4. Aggregate patches into output image using IRLS
    5. Fuse in data from content image weighted by segmentation mask
    6. Transfer color from style image to output image
    7. Denoise the output image
    8. Upscale the image for next pyramid level and repeat 2-8
    """
    # Transfer color and get segmentation mask in 3 channels
    colored_content = color_transfer(content, style)
    print("colored_content shape:", colored_content.shape)

    segmentation_mask = get_segmentation_mask(content, max_pixel_val=max_pixel_val)
    segmentation_mask = np.dstack([segmentation_mask] * content.shape[-1])
    print("segmentation_mask shape:", segmentation_mask.shape)
    # Get segmentation mask and pyramids
    style_pyramid = get_pyramid(style, num_pyramid_layers)
    content_pyramid = get_pyramid(colored_content, num_pyramid_layers)
    segmentation_pyramid = get_pyramid(segmentation_mask, num_pyramid_layers)

    # Initialize with content image and heavy Gaussian noise
    noise_sigma = 50
    output_image = content_pyramid[0] + np.random.normal(scale=noise_sigma, size=content_pyramid[0].shape) 


    for L in range(num_pyramid_layers):
        print(f"Running on layer {L}")
        scaled_style = style_pyramid[L]
        scaled_content = content_pyramid[L]
        scaled_segmentation = segmentation_pyramid[L]

        for patch_size, patch_spacing in zip(patch_sizes, patch_spacings):
            # style_patches, pca, nn = pca_nn.get_patches_pca_nn(scaled_style, patch_size, patch_spacing=patch_spacing)
            # print("style_patches shape:", style_patches.shape)
            # print("nn shape:", nn.shape)

            # from github: new patch matching and robust aggregation
            ############ - begin github additions
            style_patches = extract_patches(scaled_style, patch_shape=(patch_size, patch_size, 3), extraction_step=patch_spacing)
            style_patches = style_patches.reshape(-1, patch_size * patch_size * 3)
            njobs = 1
            if (L == 0) or (L == 1 and patch_size <= 13):
                njobs = -1
            projection_matrix = 0
            if patch_size <= 21:
                new_style_patches, projection_matrix = pca(style_patches)
                neighbors = NearestNeighbors(n_neighbors=1, p=2, n_jobs=njobs).fit(new_style_patches)
            else:
                neighbors = NearestNeighbors(n_neighbors=1, p=2, n_jobs=njobs).fit(style_patches)
            
            style_patches = style_patches.reshape((-1, patch_size, patch_size, 3))

            for _ in range(num_iters):
                X_patches_raw = extract_patches(output_image, patch_shape=(patch_size, patch_size, 3), extraction_step=patch_spacing)
                print("X_patches_raw shape:", X_patches_raw.shape)
                # raise Exception
                for i in range(IRLS_it):
                    solve_irls(output_image, X_patches_raw, patch_size, patch_spacing, style_patches, neighbors, projection_matrix)
            ########## - end github additions
                # matches = pca_nn.get_patch_matches(output_image, scaled_style, patch_size, patch_spacing=patch_spacing, pca=pca, nn=nn, reference_patches=style_patches)
                # output_image = aggregate_patches(output_image, matches, patch_spacing, num_irls_iters=num_irls_iters)
                output_image = fuse_content(output_image, scaled_content, scaled_segmentation)    
                output_image = color_transfer(output_image, scaled_style)
                output_image = (denoise_tv_chambolle(output_image, weight=0.05) * max_pixel_val)
        
        # Upscale to match the size of the next layer content image unless last layer
        if L < num_pyramid_layers - 1:
            output_image = cv2.resize(output_image, (content_pyramid[L + 1].shape[1], content_pyramid[L + 1].shape[0]))

    return output_image



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--style", help="Path to style image.")
    parser.add_argument("-c", "--content", help="Path to content image.")
    parser.add_argument("-o", "--out", help="Path for output image.")

    args = parser.parse_args()

    style_path = os.path.join("images", "styles", args.style)
    content_path = os.path.join("images", "contents", args.content)
    out_path = os.path.join("images", "results", args.out)

    content_img = cv2.imread(content_path)
    style_img = cv2.imread(style_path)

    output_img = style_transfer(content_img, style_img)
    cv2.imwrite(out_path, output_img)
    