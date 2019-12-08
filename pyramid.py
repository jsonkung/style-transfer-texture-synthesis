import cv2
import os

def get_pyramid(img, num_layers):
    """
    Gets a Gaussian pyramid of img downscaled num_layers times

    :param img: ndarray representing image to downscale
    :param num_layers: int number of downscaled layers to obtain
    :return: return list of scaled version of img from smallest to largest
    """
    layers = [None] * num_layers
    scaled_img = img.copy()
    layers[-1] = scaled_img

    for layer in range(num_layers-2, -1, -1):
        scaled_img = cv2.pyrDown(scaled_img)
        layers[layer] = scaled_img

    return layers

if __name__ == "__main__":
    image_path = os.path.join("images", "contents", "dog.jpg")
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    layers = get_pyramid(img, 3)
    for layer in layers:
        print(layer.shape)