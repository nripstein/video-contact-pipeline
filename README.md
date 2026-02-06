
# Contact Duration Detection
This repository supports the thesis work in the <a href="https://pnb.mcmaster.ca/goldreich-lab/CurrentRes.html#Modeling">Goldreich Lab</a>, focused on estimating object‑contact duration from hand video. It contains a refactored, project‑specific pipeline built on top of the original 100‑DOH model.

For project context and results, see `docs/ug-thesis.md`.

## Installation
This was only tested on Ubuntu 22.04 with an NVIDIA GPU. CUDA and cuDNN must be installed before using this repository.


## Prerequisites

Create a conda environment, install pytorch-1.12.1, cuda-11.3:
* python=3.8
* cudatoolkit=11.3
* pytorch=1.12.1



## Preparation

Clone the repository:
```
git clone https://github.com/nripstein/Thesis-100-DOH && cd Thesis-100-DOH
```

## Environment & Compilation
### Environment Setup:
Copy and paste the following commands into the command line:
```
conda create --name handobj_new python=3.8
conda activate handobj_new
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
cd lib
# then install gcc 10  
sudo apt install gcc-10 g++-10
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
# now that we have gcc 10, can compile
CC=/usr/bin/gcc-10 CXX=/usr/bin/g++-10 python setup.py build develop
```
Install all the python dependencies using pip:
```
pip install -r requirements.txt
```

<!-- You will meet some errors about coco dataset: (not the best but the easiest solution)
```
cd data
git clone https://github.com/pdollar/coco.git 
cd coco/PythonAPI
make
``` -->
<!-- 
If you meet some error about spicy, make sure you downgrade to scipy=1.1.0:
```
pip install scipy=1.1.0
``` -->

## New Pipeline (Batch Capable)

See `QUICKSTART.md` for a concise, runnable guide.

### How to run

Preprocess only:
```
python run_pipeline.py --input /path/to/input --output-dir results/run/ --preprocess-only --no-crop --no-flip
```

Inference + condense:
```
python run_pipeline.py --input /path/to/input --output-dir results/run/
```

Batch mode:
```
python run_pipeline.py --input /path/to/folder_of_videos --output-dir results/batch_run/
```

Batch outputs are written to:
```
results/batch_run/<video_stem>/
```

Optional GT barcode overlay:
```
python run_pipeline.py --input /path/to/input --output-dir results/run/ --gt-csv /path/to/gt.csv
```

Regression testing:
```
pytest -q
TEST_INPUT="/path/to/input" scripts/run_parity_suite.sh
```

Legacy code lives under `archive/` (reference‑only).

### Metrics (quantitative evaluation)

Evaluate condensed predictions against a GT CSV:
```
python scripts/evaluate_metrics.py --pred /path/to/detections_condensed.csv --gt /path/to/gt.csv
```

This reports MoF (frame-wise accuracy), Edit score, and F1@{10,25,50,75}.

## Provenance
This repository is adapted from the [100‑DOH Repository](https://github.com/ddshan/hand_object_detector),
the code for *Understanding Human Hands in Contact at Internet Scale* (CVPR 2020, Oral).
Dandan Shan, Jiaqi Geng*, Michelle Shu*, David F. Fouhey.
Project and dataset webpage: http://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/
