import struct
import numpy as np

def decode_mnsit_image(idx3_ubyte_file):
    bin_data = open(idx3_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows * num_cols))
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset), dtype='float64')
        offset += struct.calcsize(fmt_image)
    return images / 255.
def msnit_load():
    f = open('./train-labels.idx1-ubyte', 'rb')
    f.read(4)  # skip magic
    c = f.read(4)
    num = int.from_bytes(c, 'big')

    labels = np.empty((num, 10), dtype='float64')
    for x in range(num):
        labels[x][(int.from_bytes(f.read(1), 'big'))] = 1.

    f.close()

    imgs = decode_mnsit_image('./train-images.idx3-ubyte')

    return labels, imgs 
     