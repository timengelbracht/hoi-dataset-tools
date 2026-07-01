#!/usr/bin/env python3
from projectaria_tools.utils.vrs_to_mp4_utils import convert_vrs_to_mp4


input_vrs = "/bags/dlab_testing_2.vrs"  # ← CHANGE THIS
output_video = "/bags/aria_rgb.mp4"

convert_vrs_to_mp4(input_vrs, output_video, '/temp', 1)