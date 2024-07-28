import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

class AsciiArt:
    def __init__(self, image_filename: str, texture_filename: str, ) -> None:
        self.image_filename: str = image_filename
        self.texture_filename: str = texture_filename

        # load image, fill data
        self.image: np.ndarray = plt.imread(self.image_filename)
        self.texture_buffer: np.ndarray = plt.imread(self.texture_filename)

        # store dimensions of image
        self.height: int = self.image.shape[0]
        self.width: int = self.image.shape[1]

        self.get_luminance()
        self.reduce_texture_dim()

        # ascii character size (assuming it to be square)
        self.char_size: int = self.texture_buffer.shape[0]

    def reduce_texture_dim(self) -> None:
        # reduce the dimension of texture from (height, width, 4) to (height, width)
        self.texture_buffer = self.texture_buffer[:, :, 0]

    # convert image to grey scale
    def get_luminance(self) -> None:
        # check if we need to normalize pixel value
        if self.image.max() > 1:
            self.image = self.image / 255.0
        # TODO: this might need to be changed to standard format of converting RGB to greyscale
        r, g, b = self.image[:, :, 0], self.image[:, :, 1], self.image[:, :, 2]
        self.image = 0.2989 * r + 0.5870 * g + 0.1140 * b
    
    # downsample image by given factor
    def downsample(self, factor: int) -> None:
        buffer: np.ndarray = np.zeros((self.height // factor, self.width // factor))

        for i in range(buffer.shape[0]):
            for j in range(buffer.shape[1]):
                buffer[i, j] = self.image[i * factor, j * factor]
        
        # store buffer into image
        self.image = buffer
        self.height = buffer.shape[0]
        self.width = buffer.shape[1]

    # quantize image into given count
    def quantize(self, count: int) -> None:
        self.image = np.floor(self.image * count) / float(count)

    def get_char(self, index: int) -> np.ndarray:
        return self.texture_buffer[:, index * self.char_size: index * self.char_size + self.char_size]
    
    # sobel operator
    def sobel(self) -> None:
        self.gx = ndimage.sobel(self.image, 0) # horizontal
        self.gy = ndimage.sobel(self.image, 1) # vertical
    
    def get_magnitude(self, threshold: float) -> None:
        self.magnitude = np.sqrt(self.gx**2 + self.gy**2)

        # threshold
        self.magnitude[self.magnitude < threshold] = 0

    def find_angle(self) -> None:
        self.theta = np.arctan2(self.gy, self.gx)
        self.abs_theta = np.abs(self.theta) / np.pi

    def edge_quantize(self) -> None:
        # convert all data to 0 to 0.3
        buffer: np.ndarray = np.zeros(self.image.shape)
        for i in range(self.height):
            for j in range(self.width):
                if self.abs_theta[i, j] <= 0.2:
                    self.abs_theta[i, j] = 0
                    self.theta[i, j] = 0
                direction = -1
                if not (self.gx[i,j] == 0 and self.gy[i, j] == 0):
                    if 0.0 <= self.abs_theta[i, j] and self.abs_theta[i, j] < 0.05:
                        direction = 1
                    elif 0.9 < self.abs_theta[i, j] and self.abs_theta[i, j] <= 1.0:
                        direction = 1
                    elif 0.45 < self.abs_theta[i, j] and self.abs_theta[i, j] < 0.55:
                        direction = 0
                    elif 0.05 < self.abs_theta[i, j] and self.abs_theta[i, j] < 0.45:
                        if self.theta[i, j] > 0:
                            direction = 2
                        else:
                            direction = 3
                    elif 0.55 < self.abs_theta[i, j] and self.abs_theta[i, j] < 0.9:
                        if self.theta[i, j] > 0:
                            direction = 3
                        else:
                            direction = 2
                buffer[i, j] = direction
        
        # store temp buffer into image and update dimensions
        self.image = buffer

    # convert image to ascii art
    # we expect image to be quantized into `len` partitions first
    def to_ascii_art(self, edge_mode: bool) -> None:
        buffer: np.ndarray = np.zeros((self.height * self.char_size, self.width * self.char_size))
        for i in range(self.height):
            for j in range(self.width):
                try:
                    if edge_mode:
                        buffer[i * self.char_size: i * self.char_size + self.char_size, j * self.char_size: j * self.char_size + self.char_size] = self.get_char(int(self.image[i, j]) + 1)
                    else:
                        buffer[i * self.char_size: i * self.char_size + self.char_size, j * self.char_size: j * self.char_size + self.char_size] = self.get_char(int(self.image[i, j] * 10))
                except:
                    continue

        # store temp buffer into image and update dimensions
        self.image = buffer
        self.height = buffer.shape[0]
        self.width = buffer.shape[1]

    # find element with max frequency
    def max_freq(self, arr: list) -> tuple[float, int]:
        arr = [i for i in arr if i != -1] # strip out all -1

        if len(arr) == 0:
            return (-1, 0)

        max = 0
        res = arr[0]
        for i in arr:
            freq = arr.count(i)
            if freq > max:
                max = freq
                res = i
        return (res, max)

    # controlled downscaling of a qunatized image
    def controlled_downsample(self, factor: int, threshold: int) -> None:
        buffer: np.ndarray = np.zeros((self.height // factor, self.width // factor))

        for i in range(0, self.height, factor):
            for j in range(0, self.width, factor):
                try:
                    max_freq_char, freq = self.max_freq(self.image[i:i + self.char_size, j:j + self.char_size].flatten().tolist())
                    if freq > threshold:
                        buffer[i // self.char_size, j // self.char_size] = max_freq_char
                    else:
                        buffer[i // self.char_size, j // self.char_size] = -1
                except:
                    continue
        
        # store buffer into image
        self.image = buffer
        self.height = buffer.shape[0]
        self.width = buffer.shape[1]

    # difference of gaussian
    def difference_of_gaussian(self, sigma: float, scale: float, tau: float, threshold: float) -> None:
        temp = ndimage.gaussian_filter(self.image, sigma)
        temp2 = ndimage.gaussian_filter(self.image, scale * sigma)

        self.image = (1 + tau) * temp - tau * temp2
        self.image[self.image >= threshold] = 1
        self.image[self.image < threshold] = 0

    def combine(self, edge_data: "AsciiArt") -> None:
        # go through each 8x8 chunks
        # if edge exist, then assign it to main image

        # assuming self.image and edge_data.image has same size
        for i in range(0, self.image.shape[0], self.char_size):
            for j in range(0, self.image.shape[1], self.char_size):
                edge_chunk = edge_data.image[i: i + self.char_size, j: j + self.char_size]
                
                # edge exist
                if edge_chunk.flatten().sum() != 0:
                    self.image[i: i + self.char_size, j: j + self.char_size] = edge_chunk

    # store bloom data before converting to ascii
    def get_bloom_data(self, threshold: float, stdev: float) -> None:
        self.bloom_data = (self.image > threshold) * self.image

        # blur bloom data
        self.bloom_data = ndimage.gaussian_filter(self.bloom_data, stdev)

        # upscale bloom data


    # after converting to ascii
    def add_bloom_data(self) -> None:
        # make sure both have same shape
        self.bloom_data = self.bloom_data[:self.image.shape[0], :self.image.shape[1]]
        self.image += self.bloom_data

    # store image to given path
    def store_image(self, outputfile: str) -> None:
        plt.imsave(outputfile, self.image, cmap = 'grey')