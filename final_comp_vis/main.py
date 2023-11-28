import numpy as np
from scipy.signal import correlate2d
import matplotlib.pyplot as plt
import imageio
from skimage.color import rgb2gray
from skimage.transform import rotate, resize
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from fast_feature_squares import where_are_pieces

import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import uuid
global global_bestGuess
global_bestGuess = 0
global global_bestGuess_file
global_bestGuess_file = ''
from skimage.feature import canny

def regioning(imgpart, squares, threshold=3):
    regionedimg = np.zeros((imgpart.shape[0], imgpart.shape[1]))
    squares = np.asarray(squares)
    meanr = np.mean(squares[:, :, :, 0], axis=0)
    meang = np.mean(squares[:, :, :, 1], axis=0)
    meanb = np.mean(squares[:, :, :, 2], axis=0)
    sigmaSqr = np.var(squares[:, :, :, 0], axis=0)
    sigmaSqg = np.var(squares[:, :, :, 1], axis=0)
    sigmaSqb = np.var(squares[:, :, :, 2], axis=0)
    for row in range(regionedimg.shape[0]):
        for col in range(regionedimg.shape[1]):
            count = 0
            red = ((imgpart[row, col, 0] - meanr[row, col]) ** 2) / sigmaSqr[row, col]
            green = ((imgpart[row, col, 1] - meang[row, col]) ** 2) / sigmaSqg[row, col]
            blue = ((imgpart[row, col, 2] - meanb[row, col]) ** 2) / sigmaSqb[row, col]
            if red > (threshold ** 2):
                count += 1
            if blue > (threshold ** 2):
                count += 1
            if green > (threshold ** 2):
                count += 1
            if count >= 1:
                regionedimg[row, col] = 1
    regionedimg = ndimage.binary_closing(regionedimg, structure=np.ones((3, 3)))
    closedIm = np.zeros(imgpart.shape)
    for row in range(4, regionedimg.shape[0] - 4):
        for col in range(4, regionedimg.shape[1] - 4):
            if regionedimg[row, col] == 1:
                closedIm[row, col, :] = imgpart[row, col, :]

    return closedIm

def show_image(file_path):
    # Read the image
    #img = mpimg.imread(file_path)
    img = file_path
    # Display the image
    plt.imshow(img,cmap='gray')
    plt.axis('off')  # Turn off axis labels
    plt.show()
from skimage.feature import canny

def sobel_mag(image):

    sobel_x = ndimage.sobel(image, axis=0)
    sobel_y = ndimage.sobel(image, axis=1)
    sobel_magnitude = sobel_x + sobel_y

    return sobel_magnitude


def normalized_cross_correlation_with_overlay(template, image, edge_threshold=0.5, alpha=0.5):
    # rotation_angles = np.arange(0, 360, 90)  # Adjust the step size as needed

    max_corr_values = []

    # for angle in rotation_angles:
    # rotated_template = rotate(template, angle, preserve_range=True)

    # rotated_template = gaussian_filter(rotated_template, sigma=1)  # Adjust sigma as needed
    # image = gaussian_filter(image, sigma=1)

    # Apply Canny edge detection
    # edges_template = canny(rotated_template).astype(np.float32)
    # edges_image = canny(image).astype(np.float32)

    # Apply threshold to the edges
    # edges_template[edges_template < edge_threshold] = 0
    # edges_image[edges_image < edge_threshold] = 0

    # Overlay edges on top of the original images

    # overlay_template = alpha * edges_template + (1 - alpha) * template
    # overlay_image = alpha * edges_image + (1 - alpha) * image

    # show_image(overlay_image)
    # show_image(overlay_template)
    show_image(image)

    cross_correlation = correlate2d(image, template, mode='same')

    template_norm = np.sqrt(np.sum(template ** 2))
    image_norm = np.sqrt(np.sum(image ** 2))

    ncc = np.max(cross_correlation) / (template_norm * image_norm)

    max_corr_values.append(ncc)

    # Return the best NCC value
    best_ncc_value = np.max(max_corr_values)
    return best_ncc_value


def compare_similarity_rotated(template, image1, fi):
    global global_bestGuess  # Add this line to indicate the use of the global variable
    global global_bestGuess_file
    ncc_result_best_rotated = normalized_cross_correlation_with_overlay(template, image1)

    max_corr_value_best_rotated = np.max(ncc_result_best_rotated)
    if max_corr_value_best_rotated > global_bestGuess:
        # print(ncc_result_best_rotated)
        global_bestGuess = max_corr_value_best_rotated
        global_bestGuess_file = fi


def resize_image(image_path, target_size):
    # Read the image
    image = imageio.imread(image_path)

    # Resize the image
    resized_image = ndimage.zoom(image, (target_size / image.shape[0], target_size / image.shape[1], 1), order=1)

    return resized_image

# Example usage:
# Assume 'template', 'image1', and 'image2' are your template and two images to compare
template = 'guess1.jpg'

piece_locations = where_are_pieces(template, piece_img_size = 128, n_star = 5)
print(piece_locations)
template_im = resize_image(template, 320)
height, width, _ = template_im.shape
num_pieces_x = width // 64
num_pieces_y = height // 64

blacksquares = []
subblack = os.path.join('data', 'black')
for filename in os.listdir(subblack):
    blacksquare = imageio.imread(os.path.join(subblack, filename))
    blacksquares.append(blacksquare)
whitesquares = []
subwhite = os.path.join('data', 'white')
for filename in os.listdir(subwhite):
    whitesquare = imageio.imread(os.path.join(subwhite, filename))
    whitesquares.append(whitesquare)
oldtem_im = template_im
template_im = rgb2gray(template_im)
show_image(template_im)
Guessed_Board = [[0 for _ in range(5)] for _ in range(5)]
y = 0
isblacksquare = False
for i in range(num_pieces_x):
    for j in range(num_pieces_y):
        left = i * 64
        upper = j * 64
        right = left + 64
        lower = upper + 64

        squares = whitesquares
        if isblacksquare:
            squares = blacksquares

        # Crop the image
        piece = oldtem_im[upper:lower, left:right]
        piece = regioning(piece, squares)
        piece = rgb2gray(piece)
        for subdirectory_name in os.listdir('regioneddata'):
            subdirectory_path = os.path.join('regioneddata', subdirectory_name)

            if os.path.isdir(subdirectory_path) and subdirectory_name != 'black' and subdirectory_name != 'white':

                # Loop through all files in the current subdirectory
                for filename in os.listdir(subdirectory_path):
                    file_path = os.path.join(subdirectory_path, filename)

                    image = resize_image(file_path, 64)
                    image = rgb2gray(image)
                    compare_similarity_rotated(piece, image, subdirectory_name)
        #Final failsafe for no guess
        if global_bestGuess_file == '' or piece_locations[j, i] == 0:
            if i % 2 == 0:
                if j % 2 == 0:
                    global_bestGuess_file = 'white'
                else:
                    global_bestGuess_file = 'black'
            else:
                if j % 2 == 0:
                    global_bestGuess_file = 'black'
                else:
                    global_bestGuess_file = 'white'
        Guessed_Board[j][i] = global_bestGuess_file
        global_bestGuess_file = ''
        global_bestGuess = 0
        y = y + 1
        print(y)
        isblacksquare = not isblacksquare
print(np.asarray(Guessed_Board))