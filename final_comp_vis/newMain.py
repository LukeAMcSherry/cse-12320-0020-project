import numpy as np
import itertools
import imageio
from scipy import ndimage
import os
import uuid
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

import matplotlib.animation as animation

def resize_image(image_path, target_height, target_width):
    # Read the image
    image = imageio.imread(image_path)

    # Resize the image
    resized_image = ndimage.zoom(image, (target_height / image.shape[0], target_width / image.shape[1], 1), order=1)

    return resized_image

def show_image(img):
    # Read the image
    #img = mpimg.imread(file_path)
    # Display the image
    plt.imshow(img)
    # plt.title(f"Gauss: {gauss}")
    plt.axis('off')  # Turn off axis labels
    plt.show()

def cut_and_save(image_path,background_image_path, output_folder):
    # Resize the image to 768x640
    resized_image = resize_image(image_path, 640, 640)

    empty_board_image = resize_image(background_image_path, 640, 640)

    gauss_animation = []
    fig = plt.figure()
    ax = fig.add_subplot(111)

    

    # titles = [f"Frame {gauss_sig}"]

    #for gauss_sig in tqdm(range(1, 200, 1)):
    image_animation = []

    for thresh_r in tqdm(range(0,255, 50)):
        for thresh_g in range(0,255, 50):
            for thresh_b in range(0,255,50):

                gauss_sig = 0.8

                empty_board_image = gaussian_filter(empty_board_image, sigma=gauss_sig)
                resized_image = gaussian_filter(resized_image, sigma=gauss_sig)


                diff_r = np.abs(resized_image[:, :, 0] - empty_board_image[:, :, 0])
                diff_g = np.abs(resized_image[:, :, 1] - empty_board_image[:, :, 1])
                diff_b = np.abs(resized_image[:, :, 2] - empty_board_image[:, :, 2])

                mask_r = diff_r < 50
                mask_g = diff_g < thresh_g
                mask_b = diff_b < thresh_b

                mask_combined = np.logical_or.reduce((mask_r, mask_g, mask_b))
                result = np.where(mask_combined[:, :, None], resized_image, 0)

                # result = np.stack([diff_r, diff_g, diff_b], axis = -1)
                img = plt.imshow(result, animated=True)

                image_animation.append([img])



    # plt.title(f"Sigma = {gauss_sig}")
    # show_image(result)
    # img = plt.imshow(result, animated=True)

        
    # gauss_animation.append([img], )

        # gauss_animation = show_image(resized_image, gauss_sig)

    # ani = animation.ArtistAnimation(fig, gauss_animation, interval= 33.333, blit=True)
    # ani.save('animated_blurred_board.mp4', writer='ffmpeg', fps=15)
    ani = animation.ArtistAnimation(fig, image_animation, interval= 33.333, blit=True)
    ani.save('animated_rgb_thresh_board.mp4', writer='ffmpeg', fps=15)

# To display the animation
# plt.show()


    # Get the dimensions of the resized image
    height, width, _ = result.shape

    # Check if the image dimensions are suitable for cutting into 128x128 pieces
    if width % 128 != 0 or height % 128 != 0:
        raise ValueError("Image dimensions are not divisible by 128.")

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Calculate the number of pieces in each dimension
    num_pieces_x = width // 128
    num_pieces_y = height // 128

    # Loop through the pieces, cut, and save
    # for i in range(num_pieces_x):
    #     for j in range(num_pieces_y):
    #         left = i * 128
    #         upper = j * 128
    #         right = left + 128
    #         lower = upper + 128

    #         # Crop the image
    #         piece = result[upper:lower, left:right, :]

    #         # Generate a random 6-letter string as the filename
    #         random_filename = str(uuid.uuid4())[:6]

    #         # Save the piece
    #         piece_path = os.path.join(output_folder, f"{random_filename}.png")
    #         imageio.imwrite(piece_path, piece)

if __name__ == "__main__":
    input_image_path = 'board3.jpg'
    background_image_path = 'background.jpg'  # Replace with the path to your original image
    output_folder_path = 'data/r0_b0_g0_back_diff_gauss0.1'  # Specify the output folder


    empty_board_image = resize_image(background_image_path, 640, 640)
    plt.imshow(empty_board_image)
    plt.show()
    # cut_and_save(input_image_path,background_image_path, output_folder_path)
