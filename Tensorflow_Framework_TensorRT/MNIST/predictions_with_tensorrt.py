from tensorflow.examples.tutorials.mnist import input_data
from tensorrt.lite import Engine
import tensorrt as trt
from PIL import Image
import numpy as np
import os
import functools
import time
import matplotlib.pyplot as plt
from random import randint
import random

PLAN_single = '/tmp/tensorflow/mnist/tf_model_batch10_fp32.engine'  # engine filename for batch size 1
PLAN_half = '/tmp/tensorflow/mnist/tf_model_batch1_fp16.engine'
IMAGE_DIR = '/tmp/tensorflow/mnist/input_data/test-images'
BATCH_SIZE = 10

# Utility Functions
# We define here a few utility functions. These functions are lists here:
# 1.Analyze the prediction
# 2.Convert a image to format that is identical to the format during training
# 3.Organize the images into a list of numpy numpy array
# 4.Time the compute time of a function

def analyze(output_data):
    LABELS = ["0","1","2","3","4","5","6","7","8","9"]
    output = output_data.reshape(-1, len(LABELS))

    top_classes = [LABELS[idx] for idx in np.argmax(output, axis=1)]
    top_classes_prob = np.amax(output, axis=1)

    return top_classes, top_classes_prob

def image_to_np_CHW(image):
    return np.asarray(
        image.resize(
            (28, 28),
            Image.ANTIALIAS
        )).transpose([1, 0]).astype(np.float32)

def load_and_preprocess_images():
    file_list = [f for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f))]
    images_trt = []
    for f in file_list:
        images_trt.append(image_to_np_CHW(Image.open(os.path.join(IMAGE_DIR, f))))

    images_trt = np.stack(images_trt)

    num_batches = int(len(images_trt) / BATCH_SIZE)

    images_trt = np.reshape(images_trt[0:num_batches * BATCH_SIZE], [
        num_batches,
        BATCH_SIZE,
        images_trt.shape[1],
        images_trt.shape[2]
    ])

    return images_trt

def timeit(func):
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        startTime = time.time()
        retargs = func(*args, **kwargs)
        elapsedTime = time.time() - startTime
        print('function [{}] finished in {} ms'.format(
            func.__name__, int(elapsedTime * 10000)))
        return retargs
    return newfunc


# Prepare tensorRT engine
def load_TRT_engine(plan):
    engine = Engine(PLAN=plan, postprocessors={"fc2/Relu":analyze})
    return engine

engine_single = load_TRT_engine(PLAN_single)
# engine_half = load_TRT_engine(PLAN_half)

# Load all the test data
images_trt = load_and_preprocess_images()

# MNIST_DATASETS = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data")
# test_data= MNIST_DATASETS.test.next_batch(10)
# # Prepare function to do inference with TensorRT
# images_trt = []
# images_trt.append(test_data)

def generate_cases(num):
    '''
    Generate a list of raw data (data will be processed in the engine) and answers to compare to
    '''
    cases = []
    labels = []
    for c in range(num):
        rand_file = randint(0, 9)
        im = Image.open(str(rand_file) + ".bmp")
        arr = np.array(im).reshape(1,28,28) #Make the image CHANNEL x HEIGHT x WIDTH
        cases.append(arr) #Append the image to list of images to process
        labels.append(rand_file) #Append the correct answer to compare later
    return cases, labels

# images_trt, target = generate_cases(10000)

@timeit
def infer_all_images_trt(engine):
    results = []
    for image in images_trt:
        result = engine.infer(image)
        # prediction = np.argmax(engine.infer(image)[0], 1).tolist()
        # pre = [a[0][0] for a in prediction]
        results.extend(result)
    return results

# DO inference with TRT
results_trt_single = infer_all_images_trt(engine_single)
# results_trt_half = infer_all_images_trt(engine_half)
print(results_trt_single)
print(len(results_trt_single))
# print(images_trt)

# Validate results
# correct = 0.0
# print ("[LABEL] | [RESULT]")
# for l in range(len(target)):
#     print ("   {}    |    {}   ".format(target[l], results_trt_single[l][0][0][0][0]))
#     if int(target[l]) == int(results_trt_single[l][0][0][0][0]):
#         correct += 1
#     else:
#         print('error')
# print ("Inference: {:.2f}% Correct".format((correct / len(target)) * 100))

for i in range(len(results_trt_single)):
    # plt.imshow(images_trt[i,0,0],  cmap='gray')
    # plt.show()
    print results_trt_single[i][0][0][0][0]
    # print results_trt_half[i][0][0][0]
