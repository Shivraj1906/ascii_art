from AsciiArt import AsciiArt

edges = AsciiArt('image7.png', 'res/edgesASCII.png')
edges.difference_of_gaussian(2.0, 1.6, 1.0, 0.2)
edges.sobel()
edges.get_magnitude(0.2)
edges.find_angle()
edges.edge_quantize()
edges.controlled_downsample(8, 12)
edges.to_ascii_art(True)
edges.store_image('output/edges7.png')

fill = AsciiArt('image7.png', 'res/fillASCII.png')
fill.downsample(8)
fill.quantize(10)
fill.to_ascii_art(False)
fill.combine(edges)
fill.store_image('output/main7.png')