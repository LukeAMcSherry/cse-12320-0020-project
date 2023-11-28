import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import io, filters, color
from tqdm import tqdm
import matplotlib.animation as animation
import os
from scipy import ndimage
import imageio

from scipy.ndimage import gaussian_filter


##################################
### FAST FEATURE HELPER FUNCTIONS
##################################


def extract_pixel_neighborhood(img, point_cords):

    y, x = point_cords

    radius_cords = [(y,x-3),(y+1,x-3), (y+2, x-2), (y+3, x-1), (y+3, x), (y+3, x+1), (y+2, x+2), (y+1, x+3), (y, x+3), (y-1, x+3), (y-2, x+2), (y-3, x+1), (y-3, x), (y-3, x-1), (y-2, x-2), (y-1, x-3)]
    values = [img[y,x] for y, x in radius_cords]

    return values


def is_feature(values, point_value, threshold, n_star):
    
    contig_arr = np.zeros((16,))

    contig_arr[values > (point_value + threshold)] = 1
    contig_arr[values < (point_value - threshold)] = -1

    # Repeating array allows for simple detection of looped sequences
    np.repeat(contig_arr, 2)

    longest_seq = 1
    for i in range(1, len(contig_arr)):
        
        if contig_arr[i-1] == contig_arr[i] and not contig_arr[i] == 0:
            longest_seq += 1
        else:
            longest_seq = 1

    return longest_seq >= n_star


def fast_feature_points(img, threshold, n_star = 9):

    fast_features = []

    height, width = img.shape

    for x in range(3,width-3):
        for y in range(3,height-3):
            point_val = img[y,x]

            neighborhood_vals = extract_pixel_neighborhood(img, (y,x))

            if is_feature(neighborhood_vals, point_val, threshold, n_star):
                fast_features.append((y,x))

    return np.array(fast_features)

##################################
### IMAGE HELPER FUNCTIONS
##################################

def resize_image(image_path, target_height, target_width):
    # Read the image
    image = imageio.imread(image_path)

    # Resize the image
    resized_image = ndimage.zoom(image, (target_height / image.shape[0], target_width / image.shape[1], 1), order=1)

    return resized_image

##################################
### HELPER FUNCTIONS
##################################

def animations(file_path, ani_name, n_star = 5):

        feature_point_imgs = []
        fig = plt.figure()

        title_text = plt.text(0.5, 0.9, "", ha='center', va='bottom', transform=fig.transFigure)

        for threshold in tqdm(range(0,40, 1)):

            threshold /= 100

            img, feature_cords = single_image_feature_points(file_path, threshold = threshold , n_star=n_star)

            ani_img = None
            if len(feature_cords) == 0:
                ani_img = plt.imshow(img, cmap = "gray", animated = True)
            else:
                plt.imshow(img, cmap = "gray", animated=True)
                ani_img = plt.scatter(feature_cords[:,1], feature_cords[:,0], s = 5, c = "Red")
            
            title_text.set_text(f"Fast Feature Points \nThreshold {threshold}")

            feature_point_imgs.append([ani_img, title_text])

        ani = animation.ArtistAnimation(fig, feature_point_imgs, interval= 33.333, blit=True)
        ani.save(f'{ani_name}.mp4', writer='ffmpeg', fps=10)


def mod_feature_points(image, threshold, sigma, n_star = 9):

    y, x = image.shape
    cropped_img = image[7: y-7, 7:x - 7]
    cropped_img = ndimage.sobel(cropped_img, axis=1)
    cropped_img = gaussian_filter(cropped_img, sigma = sigma)

    feature_cords = fast_feature_points(cropped_img, threshold = threshold, n_star  = n_star)
    
    return feature_cords


def single_image_feature_points(image_path, threshold, sigma=1, size = 128, n_star=9):

    image = resize_image(image_path, size, size)
    image = color.rgb2gray(image)

    y, x = image.shape
    cropped_img = image[3: y-3, 3:x - 3]
    cropped_img = ndimage.sobel(cropped_img, axis=1)
    cropped_img = gaussian_filter(cropped_img, sigma = sigma)

    feature_cords = fast_feature_points(cropped_img, threshold = threshold, n_star= n_star)
    num_features = len(feature_cords)

    # plt.imshow(cropped_img, cmap = "gray", animated=True, vmax = 0.5, vmin = -0.5)
    # plt.colorbar()
    # output_img = plt.scatter(feature_cords[:,1], feature_cords[:,0], s = 5, c = "Red")
    # plt.title(f"Sigma : {sigma}, Thresh : {threshold}, Num Features : {num_features}")
    #plt.show()

    return cropped_img, feature_cords


##################################
### MAIN
##################################

def where_are_pieces(board_filepath, piece_img_size = 128, n_star = 9):
    
    # Get the dimensions of the resized image
    board_img = resize_image(board_filepath, piece_img_size*5, piece_img_size*5)
    height, width, _ = board_img.shape

    # Check if the image dimensions are suitable for cutting into 128x128 pieces
    if width % piece_img_size != 0 or height % piece_img_size != 0:
        raise ValueError(f"Image dimensions are not divisible by {piece_img_size}.")

    # Calculate the number of pieces in each dimension
    num_pieces_x = width // piece_img_size
    num_pieces_y = height // piece_img_size


    num_features = np.zeros((5,5))
    is_piece = np.zeros((5,5))

    # Loop through the pieces, cut, and save
    for i in tqdm(range(num_pieces_x)):
        for j in range(num_pieces_y):
            left = i * piece_img_size
            upper = j * piece_img_size
            right = left + piece_img_size
            lower = upper + piece_img_size

            # Crop the image
            piece = board_img[upper:lower, left:right, :]
            bw_piece = color.rgb2gray(piece)
            
            piece_idx = (i*5 + j) 

            # White Pieces
            if piece_idx % 2 == 0:
                # Change to num features in image
                features = mod_feature_points(bw_piece, threshold=0.17 , sigma= 0.3, n_star = n_star)
                num_points = len(features)
                num_features[j,i] = num_points

                if num_points > 40:
                    is_piece[j,i] = 1

            # Black Pieces
            else:
                features = mod_feature_points(bw_piece, threshold=0.17 , sigma=1, n_star = n_star)
                num_points = len(features)
                num_features[j,i] = num_points

                if num_points > 40:
                    is_piece[j,i] = 1

    # return matrix of where pieces are
    return is_piece

def where_are_pieces_mean(board_img, piece_img_size = 128):


    board = resize_image(board_img, piece_img_size*5, piece_img_size*5)
    board = gaussian_filter(board, sigma = 2)
    bw_board = color.rgb2gray(board)

    background = resize_image("background.jpg", 640, 640)
    background = gaussian_filter(background, sigma = 2)
    bw_background = color.rgb2gray(background)

    board_img = np.abs(bw_board - bw_background)

    height, width = board_img.shape

    # Check if the image dimensions are suitable for cutting into 128x128 pieces
    if width % piece_img_size != 0 or height % piece_img_size != 0:
        raise ValueError(f"Image dimensions are not divisible by {piece_img_size}.")

    # Calculate the number of pieces in each dimension
    num_pieces_x = width // piece_img_size
    num_pieces_y = height // piece_img_size


    tile_mean = np.zeros((5,5))

    # Loop through the pieces, cut, and save
    for i in tqdm(range(num_pieces_x)):
        for j in range(num_pieces_y):
            left = i * piece_img_size
            upper = j * piece_img_size
            right = left + piece_img_size
            lower = upper + piece_img_size

            # Crop the image
            piece = board_img[upper:lower, left:right]
            bw_piece = color.rgb2gray(piece)


            # fig, axs = plt.subplots(2, 1)
            
            # axs[0].hist(bw_piece.flatten())
            # axs[1].imshow(bw_piece, cmap = "gray")
            # plt.show()

            tile_mean[j,i] = (bw_piece > 0.1).sum() #(bw_piece > 0.10).sum()
        
    return tile_mean > 3000
            


if __name__ == '__main__':

    # single_image_feature_points("data/blackpawn/8c0c4c.png", threshold = 0.17, sigma = 1, size = 128)
    # single_image_feature_points("/Users/robert/Desktop/final/data/black/773df4.png", threshold = 0.17, sigma = 1, size = 128)
    # single_image_feature_points("data/blackking/bd5fa1.png", threshold = 0.17, sigma = 1, size = 128)
    # animations("data/blackking/bd5fa1.png", "black_king_fast_animation")

    where_are_pieces("guess1.jpg", piece_img_size = 128, n_star = 5)
    # print(where_are_pieces_mean("guess1.jpg", piece_img_size = 128))

    # board = resize_image("board2.jpg", 640, 640)
    # #board = gaussian_filter(board, sigma = 2)
    # bw_board = color.rgb2gray(board)

    # background = resize_image("background.jpg", 640, 640)
    # #background = gaussian_filter(background, sigma = 2)
    # bw_background = color.rgb2gray(background)

    # print(where_are_pieces_mean(np.abs(bw_board - bw_background)))

    # plt.imshow(np.abs(bw_board - bw_background), cmap = "gray")
    # plt.title("Background Subtraction")
    # plt.colorbar()
    # plt.show()

    

