import struct
import numpy as np
import  matplotlib.pyplot as plt
from PIL import Image
filename='/tmp/tensorflow/mnist/input_data/t10k-images-idx3-ubyte'
binfile=open(filename,'rb')
buf=binfile.read()

index=0
magic,numImages,numRows,numColumns=struct.unpack_from('>IIII',buf,index)
index+=struct.calcsize('>IIII')
for image in range(0,numImages):
    im=struct.unpack_from('>784B',buf,index)
    index+=struct.calcsize('>784B')

    im=np.array(im,dtype='uint8')
    im=im.reshape(28,28)
   # fig=plt.figure()
   # plotwindow=fig.add_subplot(111)
   # plt.imshow(im,cmap='gray')
   # plt.show()
    im=Image.fromarray(im)
    im.save('/tmp/tensorflow/mnist/input_data/test-images/test_%s.bmp'%image,'bmp')


import numpy as np
import struct
import matplotlib.pyplot as plt


train_images_idx3_ubyte_file = '/tmp/tensorflow/mnist/input_data/train-images-idx3-ubyte'

train_labels_idx1_ubyte_file = '/tmp/tensorflow/mnist/input_data/train-images-idx3-ubyte'

test_images_idx3_ubyte_file = '/tmp/tensorflow/mnist/input_data/t10k-images-idx3-ubyte'

test_labels_idx1_ubyte_file = '/tmp/tensorflow/mnist/input_data/t10k-labels-idx3-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file):

    bin_data = open(idx3_ubyte_file, 'rb').read()


    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print '魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols)

    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print '已解析 %d' % (i + 1) + '张'
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
        images[i] = Image.fromarray(images[i])
        images[i].save('/tmp/tensorflow/mnist/input_data/test-images/train_%s.bmp' % i, 'bmp')
    return images


def decode_idx1_ubyte(idx1_ubyte_file):

    bin_data = open(idx1_ubyte_file, 'rb').read()


    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print '魔数:%d, 图片数量: %d张' % (magic_number, num_images)


    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print '已解析 %d' % (i + 1) + '张'
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):

    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):

    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):

    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):

    return decode_idx1_ubyte(idx_ubyte_file)

def run():
    train_images = load_train_images()
    train_labels = load_train_labels()
    # test_images = load_test_images()
    # test_labels = load_test_labels()
    for i in range(10):
        print train_labels[i]
        plt.imshow(train_images[i], cmap='gray')
        plt.show()
    print 'done'

if __name__ == '__main__':
    run()