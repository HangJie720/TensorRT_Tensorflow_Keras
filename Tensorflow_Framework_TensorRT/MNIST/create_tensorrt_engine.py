import tensorrt as trt
import uff
from tensorrt.parsers import uffparser

# Configuration parameters
config = {
    # Training params
    "data_dir": "/tmp/tensorflow/mnist/input_data",  # datasets
    "train_batch_size": 10,  # training batch size
    "epochs": 5000,  # number of training epochs
    "num_train_samples": 50000,  # number of training examples
    "num_val_samples": 5000,  # number of test examples
    "learning_rate": 1e-4,  # learning rate

    # Where to save models (Tensorflow + TensorRT)
    "graphdef_file": "/tmp/tensorflow/mnist/tf_mnist_graphdef.pb",
    "frozen_model_file": "/tmp/tensorflow/mnist/tf_mnist_frozen_model.pb",
    "engine_save_dir": "/tmp/tensorflow/mnist/",

    # Needed for TensorRT
    "image_dim": 28,  # the image size (square images)
    "inference_batch_size": 1,  # inference batch size
    "input_layer": "Placeholder",  # name of the input tensor in the TF computational graph
    "output_layer": "fc2/Relu",  # name of the output tensorf in the TF conputational graph
    "number_classes": 10,  # number of classes in output (5)
    "precision": "fp32",  # desired precision (fp32, fp16)
}

def create_and_save_inference_engine():

    INPUT_LAYERS = [config['input_layer']]
    OUTPUT_LAYERS = [config['output_layer']]
    INFERENCE_BATCH_SIZE = config['inference_batch_size']

    INPUT_C = 1
    INPUT_H = config['image_dim']
    INPUT_W = config['image_dim']

    # Load your newly created Tensorflow frozen model and convert it to UFF
    uff_model = uff.from_tensorflow_frozen_model(config['frozen_model_file'], OUTPUT_LAYERS)

    # Now that we have a UFF model, we can generate a TensorRT engine by creating a logger for TensorRT.
    G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)

    # Create a UFF parser to parse the UFF file created from your TF Frozen model and identify the desired input and output nodes
    parser = uffparser.create_uff_parser()
    parser.register_input(INPUT_LAYERS[0], (INPUT_C, INPUT_H, INPUT_W), 0)
    parser.register_output(OUTPUT_LAYERS[0])

    # Build your TensorRT inference engine
    # This step performs (1) Tensor fusion (2) Reduced precision calibration
    # (3) Target-specific autotuning (4) Tensor memory management

    # Pass the logger, parser, the UFF model stream,
    # and some settings (max batch size and max workspace size)
    # to a utility function that will create the engine for us

    # Build your TensorRT inference engine
    if (config['precision'] == 'fp32'):
        engine = trt.utils.uff_to_trt_engine(
            G_LOGGER,
            uff_model,
            parser,
            INFERENCE_BATCH_SIZE,
            1 << 20,
            trt.infer.DataType.FLOAT
        )

    elif (config['precision'] == 'fp16'):
        engine = trt.utils.uff_to_trt_engine(
            G_LOGGER,
            uff_model,
            parser,
            INFERENCE_BATCH_SIZE,
            1 << 20,
            trt.infer.DataType.HALF
        )
    elif (config['precision'] == 'int8'):
        engine = trt.utils.uff_file_to_trt_engine(
            G_LOGGER,
            uff_model,
            parser,
            INFERENCE_BATCH_SIZE,
            1 << 20,
            trt.infer.DataType.INT8
        )

    # engine = trt.utils.uff_to_trt_engine(G_LOGGER, uff_model, parser, 1, 1 << 20)

    # Serialize TensorRT engine to a file for when you are ready to deploy your model.
    save_path = str(config['engine_save_dir']) + "tf_model_batch" \
                + str(INFERENCE_BATCH_SIZE) + "_" + str(config['precision']) + ".engine"
    trt.utils.write_engine_to_file(save_path, engine.serialize())
    print("Saved TensorRT engine to {}".format(save_path))

create_and_save_inference_engine()