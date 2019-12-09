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


def style_transfer(content, style, num_pyramid_layers=3, patch_sizes=(33, 21, 13, 9), patch_spacings=(28, 18, 8, 5), num_iters=3, num_irls_iters=10, max_pixel_val=255):
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
    colored_content = color_transfer(style, content)
    segmentation_mask = get_segmentation_mask(content, max_pixel_val=max_pixel_val)
    segmentation_mask = np.dstack([segmentation_mask] * content.shape[-1])

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
            style_patches, pca, nn = pca_nn.get_patches_pca_nn(scaled_style, patch_size, patch_spacing=patch_spacing)

            for _ in range(num_iters):
                matches = pca_nn.get_patch_matches(output_image, scaled_style, patch_size, patch_spacing=patch_spacing, pca=pca, nn=nn, reference_patches=style_patches)
                output_image = aggregate_patches(output_image, matches, patch_spacing, num_irls_iters=num_irls_iters)
                output_image = fuse_content(output_image, scaled_content, scaled_segmentation)
                output_image = color_transfer(scaled_style, output_image)
                cv2.imwrite(f"fusedL{L}P{patch_size}.jpg", output_image)
                output_image = (denoise_tv_chambolle(output_image, weight=0.05) * max_pixel_val)
            cv2.imwrite(f"output_L{L}P{patch_size}.jpg", output_image)
        
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
    