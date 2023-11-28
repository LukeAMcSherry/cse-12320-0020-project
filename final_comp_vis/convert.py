import numpy as np
from scipy.signal import correlate2d
import matplotlib.pyplot as plt
import imageio
from skimage.color import rgb2gray, gray2rgba
from skimage.transform import rotate, resize
from scipy import ndimage
from scipy.ndimage import gaussian_filter

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
                # regionedimg[row, col] = imgpart[row, col]
    regionedimg = ndimage.binary_closing(regionedimg, structure=np.ones((3, 3)))
    closedIm = np.zeros(imgpart.shape)
    for row in range(4, regionedimg.shape[0] - 4):
        for col in range(4, regionedimg.shape[1] - 4):
            if regionedimg[row, col] == 1:
                closedIm[row, col, :] = imgpart[row, col, :]

    return closedIm

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

newdir = 'regioneddata'
os.makedirs(newdir)
for subdirectory_name in os.listdir('data'):
    oldpath = os.path.join('data', subdirectory_name)
    catpath = os.path.join('catdata', subdirectory_name)
    newpath = os.path.join(newdir, subdirectory_name)
    if os.path.isdir(catpath):
        os.makedirs(newpath)

        for filename in os.listdir(os.path.join(catpath, 'black')):
            if filename != '.DS_Store':
                oldfileb = os.path.join(catpath, 'black', filename)
                newfile = os.path.join(newpath, filename)
                colimg = imageio.imread(oldfileb)
                img = colimg
                img = regioning(img, blacksquares)
                for row in range(colimg.shape[0]):
                    for col in range(colimg.shape[1]):
                        if img[row, col, 0] == 0:
                            colimg[row, col, :] = np.asarray([0, 0, 0])
                imageio.imwrite(newfile, colimg)
        for filename in os.listdir(os.path.join(catpath, 'white')):
            if filename != '.DS_Store':
                oldfilew = os.path.join(catpath, 'white', filename)
                newfile = os.path.join(newpath, filename)
                colimg = imageio.imread(oldfilew)
                img = colimg
                img = regioning(img, whitesquares)
                for row in range(colimg.shape[0]):
                    for col in range(colimg.shape[1]):
                        if img[row, col, 0] == 0:
                            colimg[row, col, :] = np.asarray([0, 0, 0])
                imageio.imwrite(newfile, colimg)