# export_micro_onnx.py
import os
import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
from tensorflow import keras
from typing import Tuple, Dict, Any
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_model(input_shape: Tuple[int, ...], num_classes: int) -> keras.Model:
    """Create a simple Keras model for demonstration."""
    inputs = keras.Input(shape=input_shape, name='input')
    x = keras.layers.Dense(64, activation='relu')(inputs)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def convert_to_onnx(keras_model: keras.Model, output_path: str) -> None:
    """
    Convert a Keras model to ONNX format.
    
    Args:
        keras_model: Trained Keras model
        output_path: Path to save the ONNX model
    """
    import tf2onnx
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Convert the model
    spec = (tf.TensorSpec((None, *keras_model.input.shape[1:]), tf.float32, name="input"),)
    output_path = os.path.abspath(output_path)
    
    model_proto, _ = tf2onnx.convert.from_keras(
        keras_model,
        input_signature=spec,
        output_path=output_path
    )
    
    logger.info(f"Model converted and saved to {output_path}")
    return model_proto

def optimize_onnx_model(input_path: str, output_path: str) -> None:
    """
    Optimize an ONNX model for inference.
    
    Args:
        input_path: Path to input ONNX model
        output_path: Path to save optimized model
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Quantize the model to int8
    quantize_dynamic(
        input_path,
        output_path,
        weight_type=QuantType.QUANT_INT8
    )
    
    logger.info(f"Optimized model saved to {output_path}")

def convert_to_tflite(keras_model: keras.Model, output_path: str) -> None:
    """
    Convert a Keras model to TensorFlow Lite format.
    
    Args:
        keras_model: Trained Keras model
        output_path: Path to save the TFLite model
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    
    # Save the model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    logger.info(f"TFLite model saved to {output_path}")

def test_onnx_model(model_path: str, input_shape: Tuple[int, ...]) -> None:
    """
    Test the ONNX model with sample input.
    
    Args:
        model_path: Path to the ONNX model
        input_shape: Shape of the input tensor (batch dimension excluded)
    """
    # Create a sample input
    input_data = np.random.random((1, *input_shape)).astype(np.float32)
    
    # Create ONNX Runtime session
    sess = ort.InferenceSession(model_path)
    
    # Get input and output names
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    
    # Run inference
    result = sess.run([output_name], {input_name: input_data})
    
    logger.info(f"ONNX model test successful. Output shape: {result[0].shape}")

def main():
    parser = argparse.ArgumentParser(description='Export Keras model to ONNX and TFLite formats')
    parser.add_argument('--input-shape', type=int, nargs='+', default=[10],
                       help='Input shape (e.g., 10 for a vector, 28 28 1 for an image)')
    parser.add_argument('--num-classes', type=int, default=5,
                       help='Number of output classes')
    parser.add_argument('--output-dir', type=str, default='exported_models',
                       help='Output directory for exported models')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Create and train a sample model
        logger.info("Creating and training sample model...")
        model = create_sample_model(tuple(args.input_shape), args.num_classes)
        
        # Generate random training data
        x_train = np.random.random((100, *args.input_shape))
        y_train = np.random.randint(0, args.num_classes, (100,))
        
        # Train the model (minimal training for demonstration)
        model.fit(x_train, y_train, epochs=1, verbose=1)
        
        # Export to ONNX
        onnx_path = os.path.join(args.output_dir, 'model.onnx')
        convert_to_onnx(model, onnx_path)
        
        # Test ONNX model
        test_onnx_model(onnx_path, tuple(args.input_shape))
        
        # Optimize ONNX model
        optimized_onnx_path = os.path.join(args.output_dir, 'model_optimized.onnx')
        optimize_onnx_model(onnx_path, optimized_onnx_path)
        
        # Export to TFLite
        tflite_path = os.path.join(args.output_dir, 'model.tflite')
        convert_to_tflite(model, tflite_path)
        
        logger.info("All exports completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during model export: {str(e)}")
        raise

if __name__ == "__main__":
    main()