#!/usr/bin/env python
# ===================================================================
# ScriptName:   SPACEtomo_config
# Purpose:      Configurations for deep learning models used in SPACEtomo.
#               More information at http://github.com/eisfabian/SPACEtomo
# Author:       Fabian Eisenstein
# Created:      2023/10/04
# Revision:     v1.0beta
# Last Change:  2023/10/04: outsourcing of settings from main SPACEtomo script
# ===================================================================

# Model specific settings (depend on how the model was trained)
# WG model (YOLOv8)
# WG_model_file = "2023_11_16_lamella_detect_400nm_yolo8.pt"
# WG_model_pix_size = 400         # nm / px
# WG_model_sidelen = 1024
# WG_model_categories = ["broken", "contaminated", "good", "thick"]
# WG_model_colors = ["red", "yellow", "green", "orange"]
# WG_model_nav_colors = [0, 2, 1, 3]

# # MM model (nnUNet)
# MM_model_script = "SPACEtomo_nnUNet.py"
# MM_model_folder = "model"
# MM_model_folds = [0, 1, 2, 3, 4]
# MM_model_pix_size = 22.83 / 10  # nm / px
# MM_model_max_runs = 1

from spacetomo.util import load_yaml

def parse_configuration(filename: str) -> dict:
    return load_yaml(filename)

# Load configuration
import os
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

configuration = parse_configuration(os.path.join(BASE_PATH, "config.yaml"))

base_model_path = configuration["core"]["base_path"]

det_configuration = configuration["detection"]
WG_model_file = os.path.join(base_model_path, det_configuration["WG_model_file"])
WG_model_pix_size = det_configuration["WG_model_pix_size"]
WG_model_sidelen = det_configuration["WG_model_sidelen"]
WG_model_categories = det_configuration["WG_model_categories"]
WG_model_colors = det_configuration["WG_model_colors"]
WG_model_nav_colors = det_configuration["WG_model_nav_colors"]

seg_configuration = configuration["segmentation"]
MM_model_script = seg_configuration["MM_model_script"]
MM_model_folder = os.path.join(base_model_path, seg_configuration["MM_model_folder"])
MM_model_folds = seg_configuration["MM_model_folds"]
MM_model_pix_size = seg_configuration["MM_model_pix_size"] / seg_configuration["MM_model_num_pix"]
MM_model_max_runs = seg_configuration["MM_model_max_runs"]

from pprint import pprint

pprint(configuration)