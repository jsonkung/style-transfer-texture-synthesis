import cv2
from skimage.segmentation import morphological_chan_vese
from skimage.filters import gaussian

def get_segmentation_mask(img, edge_th_1=100, edge_th_2=200, max_pixel_val=255, max_iter=2000):
    """
    """
    # Get segmentation with Chan Vese algorithm with edges
    greyscaled_content = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(greyscaled_content, edge_th_1, edge_th_2) / max_pixel_val
    segmentation = morphological_chan_vese(greyscaled_content, max_iter)
    return gaussian(edges + segmentation)