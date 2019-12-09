import numpy as np

def fuse_content(output, content, segmentation_mask):
    return ((output + content * segmentation_mask) / (segmentation_mask + 1)).astype(np.uint8)