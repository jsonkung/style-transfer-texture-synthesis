import argparse
import cv2
import os
import numpy as np

import pca_nn
from pyramid import get_pyramid
from irls import aggregate_patches

def style_transfer(content, style, num_pyramid_layers=3, patch_sizes=(33, 21, 13, 9), num_iters=3, num_irls_iters=10, patch_spacing=10):
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
    # Get segmentation mask and pyramids
    # segmentation_mask = get_segmentation_mask(content)
    style_pyramid = get_pyramid(style, num_pyramid_layers)
    content_pyramid = get_pyramid(content, num_pyramid_layers)
    # segmentation_pyramid = get_pyramid(content, num_pyramid_layers)

    # Initialize with content image and heavy Gaussian noise
    noise_sigma = 50
    output_image = content_pyramid[0] + np.random.normal(scale=noise_sigma, size=content_pyramid[0].shape) 


    for L in range(num_pyramid_layers):
        scaled_style = style_pyramid[L]
        scaled_content = content_pyramid[L]
        # scaled_segmentation = segmentation_pyramid[L]

        for patch_size in patch_sizes:
            style_patches, pca, nn = pca_nn.get_patches_pca_nn(scaled_style, patch_size, patch_spacing=patch_spacing)
            
            for _ in range(num_iters):
                matches, distances = pca_nn.get_patch_matches(output_image, scaled_style, patch_size, patch_spacing=patch_spacing, pca=pca, nn=nn, reference_patches=style_patches)
                output_image = aggregate_patches(output_image, matches, distances, patch_spacing, num_irls_iters=num_irls_iters)
                output_image = fuse_content(output_image, scaled_content, scaled_segmentation)
                output_image = transfer_color(scaled_style, output_image)
                output_image = denoise(output_image)
        
        # Upscale to match the size of the next layer content image unless last layer
        if L < num_pyramid_layers - 1:
            output_image = cv2.resize(output_image, (scaled_content[L + 1].shape[1], scaled_content[L + 1].shape[0]))

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

    content_img = cv2.imread(content_path, cv2.IMREAD_COLOR)
    style_img = cv2.imread(style_path, cv2.IMREAD_COLOR)

    output_img = style_transfer(content_img, style_img)
    