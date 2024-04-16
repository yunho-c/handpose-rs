#!/bin/bash

# Run this from project root!

# Define input/output paths
PALM_INPUT_ONNX_PATH="./model/model/palm_detection/palm_detection_full_inf_post_192x192.onnx"
LDMK_INPUT_ONNX_PATH="./model/model/hand_landmark/hand_landmark_sparse_Nx3x224x224.onnx"

PALM_OUTPUT_ORT_PATH="./model"
LDMK_OUTPUT_ORT_PATH="./model"

py -3.11 -m onnxruntime.tools.convert_onnx_models_to_ort $PALM_INPUT_ONNX_PATH --output_dir $PALM_OUTPUT_ORT_PATH --optimization_style Fixed
py -3.11 -m onnxruntime.tools.convert_onnx_models_to_ort $LDMK_INPUT_ONNX_PATH --output_dir $LDMK_OUTPUT_ORT_PATH --optimization_style Fixed