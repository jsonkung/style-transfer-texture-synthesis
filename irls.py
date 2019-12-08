import numpy as np
import cv2
from sklearn.feature_extraction.image import extract_patches

R = 0.8
NOISE = 0.000001

def aggregate_patches(output_image, matches, distances, patch_spacing, num_irls_iters=10):
    patch_size = matches.shape[1]
    num_channels = matches.shape[-1]
    weights = np.power(distances, R - 2)

    # Create aggregate matrix and reweight matrix with views to make updating easier
    aggregate_result = np.zeros(output_image.shape)
    result_patches = extract_patches(aggregate_result, patch_shape=matches.shape[1:], extraction_step=patch_spacing)
    weight_mask = np.zeros(output_image.shape, dtype=np.float32)
    weight_patches = extract_patches(weight_mask, patch_shape=matches.shape[1:], extraction_step=patch_spacing)

    reshaped_matches = matches.reshape(result_patches.shape)

    for _ in range(num_irls_iters):
        count = 0
        for i in range(result_patches.shape[0]):
            for j in range(result_patches.shape[1]):
                weight = weights[count, 0]
                result_patches[i,j] += reshaped_matches[i, j] * weight
                weight_patches[i,j] += weight
                count += 1
        result_patches /= (weight_patches + NOISE)
    return aggregate_result