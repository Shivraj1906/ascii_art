from AsciiArt import AsciiArt

sigma = 2.0
scale = 1.6
tau = 1.0
threshold = 0.3

magnitude_threshold = 0.2

char_size = 8 # dimension of a single character texture (8x8)
downsample_threshold = 12

bloom_threshold = 0.8
bloom_sigma = 50

quantize_size = 10

image_path = 'images/sample_resize.png'
image_name = image_path.split('/')[-1].split('.')[0]
edges_texture = 'res/edgesASCII.png'
fill_texture = 'res/fillASCII.png'

# edge detection part
edges = AsciiArt(image_path, edges_texture)
# preprocessor for sobel filter to reduce noise
# reference: https://users.cs.northwestern.edu/~sco590/winnemoeller-cag2012.pdf
edges.difference_of_gaussian(sigma, scale, tau, threshold) 

# apply sobel filter and find angle based on gradient
edges.sobel()
edges.get_magnitude(magnitude_threshold)
edges.find_angle()

# quantize image
edges.edge_quantize()
edges.controlled_downsample(char_size, downsample_threshold)

# convert edges to characters (slashes)
edges.to_ascii_art(True) # edge_mode = True

# store edge data (optional)
edges.store_image(f'output/{image_name}_edges.png')

# fill part
fill = AsciiArt(image_path, fill_texture)

# threshold image and then apply gaussian blur (optional)
fill.get_bloom_data(bloom_threshold, bloom_sigma)

# downsample image
fill.downsample(char_size)

# quantize into 10 partitions (since we have 10 ASCII characters)
fill.quantize(quantize_size)

# convert image to ASCII art
fill.to_ascii_art(False) # edge_mode = false

# combine edges and fill data into final output
fill.combine(edges)

# add optional bloom data
fill.add_bloom_data()

# store output
fill.store_image(f'output/{image_name}_final_with_bloom.png')