import numpy as np
import cv2
import os

from sklearn.feature_extraction.image import extract_patches
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


def get_patch_matches(source_img, reference_img, patch_size, patch_spacing=10, pca=None, nn=None, reference_patches=None):
    """
    Given a source image, reference image, patch size, and patch spacing,
    finds the closest matches to source patches of size (patch_size x patch_size)
    in reference image with patches spaced out by patch spacing

    :param source_img: ndarray representing image we want to match
    :param reference_img: ndarray representing image we want to get patch matches from
    :param patch_size: int representing the height and width of the patch
    :param patch_spacing: int representing how far apart the centers of each patch should be from each other
    :param pca: PCA object to perform PCA on patches
    :param nn: NearestNeighbor object to perform NN with source patches
    :param reference_patches: patches of reference image
    :returns: ndarray of patch matches found in reference image of size (num_patches, patch_size, patch_size, num_channels)
    """
    # Get objects for PCA transform and NN matching if not provided
    if (pca is None) or (nn is None) or (references_patches is None):
        reference_patches, pca, nn = get_patches_pca_nn(reference_img, patch_size, patch_spacing)

    # Get patches from source image and transform into PCA domain
    source_patches = extract_patches(source_img, patch_shape=(patch_size, patch_size, source_img.shape[-1]), extraction_step=patch_spacing)
    source_patches = source_patches.reshape(-1, patch_size * patch_size * source_img.shape[-1])
    pca_source_patches = pca.transform(source_patches)
    
    # Find indexes of nearest neighbors for each patch and get the corresponding raw patches from reference
    _, idxs = nn.kneighbors(pca_source_patches)
    matches = reference_patches[idxs]

    return matches.reshape(-1, patch_size, patch_size, source_img[-1])


def get_patches_pca_nn(img, patch_size, patch_spacing=10):
    """
    Given an image, patch size, and patch spacing, returns the patches of image
    based on spacing (ndarray of shape (M, n^2, c)), PCA object to transform data,
    and nearest neighbor object to find knn
    
    :param img: ndarray representing image to fit PCA and NN to
    :param patch_size: int height and width of patches to extract from img
    :param patch_spacing: int representing how far apart the centrs of each patch should be from each other
    :returns: ndarray of raw patches with shape (num_patches, patch_size * patch_size, num_channels),
        PCA object fit to img patches, and NN fit to img patches
    """
    # Get patches matrix from style image
    patches = extract_patches(img, patch_shape=(patch_size, patch_size, img.shape[-1]), extraction_step=patch_spacing)
    patches_flattened = patches.reshape(-1, patch_size * patch_size * img.shape[-1])

    # Normalize data and create PCA object fitted to patches
    patches_normalized = patches_flattened - np.mean(patches_flattened, axis=0)
    pca = PCA(n_components=0.95, svd_solver="full")
    pca_patches = pca.fit_transform(patches_normalized)

    # Create nearest neighbor matcher based on PCA domain patches
    nn_matcher = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn_matcher = matcher.fit(pca_patches)
    
    return patches.reshape(-1, patch_size * patch_size, img.shape[-1]), pca, nn_matcher
    
    

if __name__ == "__main__":
    img = cv2.imread(os.path.join("images", "contents", "dog.jpg"), cv2.IMREAD_COLOR)
    