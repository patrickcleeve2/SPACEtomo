
import os
import sys
import glob
import time
import json
import logging
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import torch
from ultralytics import YOLO, settings

# Set vars for nnUNet (not really used) before import to avoid errors
os.environ["nnUNet_raw"] = os.path.dirname(__file__)
os.environ["nnUNet_preprocessed"] = os.path.dirname(__file__)
os.environ["nnUNet_results"] = os.path.dirname(__file__)

from nnunetv2.inference import predict_from_raw_data as predict
import spacetomo.SPACEtomo_config as config

## MM model (nnUNet)
# Segmentation model
class SegmentationModel:
    def __init__(self, path: str, folds: list[int], checkpoint: str = "checkpoint_final.pth"):
        self.path = path
        self.folds = folds
        self.checkpoint = checkpoint

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu') # dynamically set device
        self.is_cuda_device = True if self.device.type == "cuda" else False 
        self.dtype=np.float16 if self.is_cuda_device else np.float32  # half precision for cuda only

        self.load_model()

    def load_model(self) -> predict.nnUNetPredictor: 
        """Loads the nnUNet model from the given path and folds."""
        predictor = predict.nnUNetPredictor(
            tile_step_size=0.5,
            perform_everything_on_gpu=self.is_cuda_device,
            device=self.device,
            allow_tqdm=False
        )

        predictor.initialize_from_trained_model_folder(
            self.path,
            self.folds,
            checkpoint_name=self.checkpoint
        )

        logging.info("Model loaded.")
        self.predictor = predictor  
        
        return self.predictor

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """"Preprocess image for nnUNet"""
        image = np.array(image, dtype=self.dtype)[np.newaxis, np.newaxis, :, :]
        return image

    def inference(self, image: np.ndarray,  out_name: str) -> np.ndarray:
        """Runs the nnUNet model on the given image and saves the result to the given output name."""
        logging.info("Starting prediction...")
        image = self.preprocess(image)

        # TODO: fix the return values, so they return mask, scores
        self.predictor.predict_single_npy_array(
            input_image=image, 
            image_properties={"spacing": [999, 1, 1]},
            output_file_truncated=out_name,
        )

        # return mask


### WG model (YOLOv8)
# Detection model
class DetectionModel:
    def __init__(self, path: str):
        self.path = path
        self.model = YOLO(path)

    def inference(self, image: np.ndarray) -> np.ndarray:
        # Do YOLO inference
        results = self.model(image)
        return results

    def preprocess_image(self, image: np.ndarray, seg_pixel_size: float, det_pixel_size: float) -> np.ndarray:
        full_shape = np.array(image.shape)
        division = np.round(full_shape * seg_pixel_size / det_pixel_size).astype(int)
        pixel_shape = full_shape // division
        pixel_map = np.zeros(division)
        for i in range(division[0]):
            for j in range(division[1]):
                pixel = image[i * pixel_shape[0]:(i + 1) * pixel_shape[0], j * pixel_shape[1]:(j + 1) * pixel_shape[1]]
                pixel_map[i, j] = np.mean(pixel)
        # Add padding to emulate grid map
        padded_map = np.zeros((config.WG_model_sidelen, config.WG_model_sidelen))
        padded_map[0: pixel_map.shape[0], 0: pixel_map.shape[1]] = pixel_map
        yolo_input = np.dstack([padded_map, padded_map, padded_map])

        return yolo_input, full_shape, pixel_shape