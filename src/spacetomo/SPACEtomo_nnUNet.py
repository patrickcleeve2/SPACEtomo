#!/usr/bin/env python
# ===================================================================
# ScriptName:   SPACEtomo_nnUnet
# Purpose:      Runs a target selection deep learning model on medium mag montages of lamella and generates a segmentation that can be used for PACEtomo target selection.
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2023/05/19
# Revision:     v1.0beta
# Last Change:  2023/11/06: changed to log output
# ===================================================================

import os
import sys
import glob
import time
import json
import logging
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from ultralytics import settings


from spacetomo.models import SegmentationModel, DetectionModel

def run_inference_pipeline(montage_file: str) -> None:
    """Runs the SPACEtomo inference pipeline on the given montage file."""
    
    # Set directory to dir of given filename
    # CUR_DIR = os.path.dirname(montage_file)
    # SPACE_DIR = os.path.dirname(__file__)

    # Start log file
    logging.basicConfig(filename=os.path.splitext(montage_file)[0] + "_SPACE.log", level=logging.INFO, format='')
    logging.info("Processing " + montage_file)

    # Import model configs (needs SPACE folder from settings file)
    # sys.path.insert(len(sys.path), SPACE_DIR)
    import spacetomo.SPACEtomo_config as config
    logging.info("Loaded config.")

    # Read segmentation classes
    with open(os.path.join(config.MM_model_folder, "dataset.json"), "r") as f:
        dataset_json = json.load(f)
    classes = dataset_json["labels"]
    logging.info("Loaded class labels.")

    # Load YOLO model
    detection_model = DetectionModel(config.WG_model_file) # load a custom model
    settings.update({'sync': False})    # no tracking by google analytics

    # Load map
    start_time = time.time()
    montage_map = np.array(Image.open(montage_file))
    time_point1 = time.time()
    logging.info("Map was loaded in " + str(int(time_point1 - start_time)) + " s.")

    # Rescale map for YOLO lamella bbox detection
    yolo_input, full_shape, pixel_shape = detection_model.preprocess_image(montage_map, 
                                                        seg_pixel_size=config.MM_model_pix_size, 
                                                        det_pixel_size=config.WG_model_pix_size)

    # Run YOLO model to detect bbox
    results = detection_model.inference(yolo_input)

    # Check and upscale resulting box
    if len(results[0].boxes) > 0:
        bbox = np.array(results[0].boxes.xyxy.to("cpu"))

        bbox[:, 0] = (bbox[:, 0]) * pixel_shape[0]
        bbox[:, 2] = (bbox[:, 2]) * pixel_shape[0]
        bbox[:, 1] = (bbox[:, 1]) * pixel_shape[1]
        bbox[:, 3] = (bbox[:, 3]) * pixel_shape[1]

        cat = np.reshape(results[0].boxes.cls.to("cpu"), (bbox.shape[0], 1))
        conf = np.reshape(results[0].boxes.conf.to("cpu"), (bbox.shape[0], 1))
        bbox = np.hstack([bbox, cat, conf])[0]

        logging.info("Lamella bounding box: " + str(bbox))
        logging.info("Lamella was categorized as: " + str(config.WG_model_categories[int(bbox[4])]) + " (" + str(round(bbox[5] * 100, 1)) + " %)")

        bounds = np.round(bbox[0:4]).astype(int)
        crop = montage_map[bounds[1]:bounds[3], bounds[0]:bounds[2]]

        time_point2 = time.time()
        logging.info("Bounding box was detected in " + str(int(time_point2 - time_point1)) + " s.")
    else:
        bounds = None
        crop = montage_map

        time_point2 = time.time()
        logging.info("WARNING: No bounding box was detected in " + str(int(time_point2 - time_point1)) + " s.")
        logging.info("Using whole montage map...")

        
    # Use temp name to pad later
    out_name = os.path.splitext(montage_file)[0] + "_seg"

    if bounds is not None:
        out_name += "temp"

    # Run segmentation model
    model = SegmentationModel(config.MM_model_folder, config.MM_model_folds)
    model.inference(crop, out_name)

    logging.info("Postprocessing...")

    # Pad segmentation to original map size
    if bounds is not None:
        segmentation = np.array(Image.open(out_name + ".png"))
        padding = ((max(0, bounds[1]), max(0, full_shape[0] - bounds[3])), (max(0, bounds[0]), max(0, full_shape[1] - bounds[2])))
        segmentation = np.pad(segmentation, padding, constant_values=classes["black"])
        seg_out = Image.fromarray(np.uint8(segmentation))
        seg_out.save(os.path.splitext(montage_file)[0] + "_seg.png")
        os.remove(out_name + ".png")

    logging.info("Prediction was completed in " + str(round((time.time() - time_point2) / 60, 1)) + " min.")
    logging.info("Finished processing " + os.path.basename(montage_file) + ".")


if __name__ == "__main__":
    # Check if filename was given
    if len(sys.argv) == 2:
        montage_file = sorted(glob.glob(sys.argv[1]))[0]
    else:
        print ("Usage: python " + sys.argv[0] + " [input]")
        sys.exit("Missing arguments!")

    # run the full inference pipeline
    run_inference_pipeline(montage_file)