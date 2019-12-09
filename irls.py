import numpy as np
import cv2
from sklearn.feature_extraction.image import extract_patches

NOISE = 0.000001

def aggregate_patches(output_image, matches, patch_spacing, num_irls_iters=10, r=0.8):
    # Ignore extract_patches deprecation warning
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    patch_size = matches.shape[1]
    num_channels = matches.shape[-1]
    

    # Create aggregate matrix and reweight matrix with views to make updating easier
    aggregate_result = output_image.copy()
    
    result_patches = extract_patches(aggregate_result, patch_shape=matches.shape[1:], extraction_step=patch_spacing)
    reshaped_matches = matches.reshape(result_patches.shape)

    for _ in range(num_irls_iters):
        
        result_patches = extract_patches(aggregate_result, patch_shape=matches.shape[1:], extraction_step=patch_spacing)
        distances = np.linalg.norm(result_patches.reshape(-1, patch_size * patch_size * num_channels) - matches.reshape(-1, patch_size * patch_size * num_channels), axis=1)
        aggregate_result[:] = 0
        weights = np.power(distances, r - 2)
        
        weight_matrix = np.zeros(output_image.shape, dtype=np.float32)
        weight_patches = extract_patches(weight_matrix, patch_shape=matches.shape[1:], extraction_step=patch_spacing)

        count = 0
        for i in range(result_patches.shape[0]):
            for j in range(result_patches.shape[1]):
                weight = weights[count]
                result_patches[i,j] += reshaped_matches[i, j] * weight
                weight_patches[i,j] += weight
                count += 1
        aggregate_result /= (weight_matrix + NOISE)

    return aggregate_result