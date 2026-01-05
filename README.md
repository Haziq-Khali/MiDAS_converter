MIDAS converter
===============

This repository contains scripts for converting RGB images to depth images using the **MiDaS model**. It is designed for pre-processing datasets for drone navigation and AI projects.

# Prerequisites
Follow the step below to setup the environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```
To convert a batch of images, use the code named Midas_batch_cvt.py 
while Midasgooglecollab.py is for testing (one image only).\

## Notes
Please change the images' directory in the source code
