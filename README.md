<div align ="center">
<h2>Vision-Language Guidance for LiDAR-based Unsupervised 3D Object Detection</h2>

[Christian Fruhwirth-Reisinger](https://scholar.google.com/citations?user=Mg5Vlp8AAAAJ&hl=de&oi=ao) <sup>1,2</sup>, [Wei Lin](https://scholar.google.com/citations?user=JJRr8c8AAAAJ&hl=de&oi=sra) <sup>3</sup>, [Dušan Malić](https://scholar.google.com/citations?user=EXovq6wAAAAJ&hl=de&oi=sra) <sup>1,2</sup>, [Horst Bischof](https://scholar.google.com/citations?user=_pq05Q4AAAAJ&hl=de&oi=ao) <sup>1</sup>, [Horst Possegger](https://scholar.google.com/citations?user=iWPrl3wAAAAJ&hl=de&oi=ao) <sup>1,2</sup>

<sup>1</sup>Graz University of Technology, <sup>2</sup>Christian Doppler Laboratory for Embedded Machine Learning, <sup>3</sup>Johannes Kepler University Linz

[![arxiv paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2408.03790)
</div>


# ViLGOD

![Overview](assets/method.png#gh-light-mode-only)
![Overview](assets/method.png#gh-dark-mode-only)

## 🚩News

`[2024-11-20]:` Code released.<br>
`[2024-09-10]:` ViLGOD has been accepted for BMVC 2024 as an oral presentation. See you in Glasgow!<br>
`[2024-08-07]:` ViLGOD [arXiv](https://arxiv.org/abs/2408.03790) paper released.<br>

## 📝 TODO List

- [x] Initial release.
- [x] Add installation details.
- [x] Add visual code run config for zero-shot detection.
- [ ] Update arXiv paper.
- [ ] Add additional run & evaluation instructions.
- [ ] Upate run scripts for multi-CPU/GPU inference.

## 🚀 Quick Start
### Tested environment
- Ubuntu 22.04
- Python 3.8
- CUDA 11.7

## Environment setup

### Creat virtual environment and intstall required packages

1) Create virtual environment

```bash
virtualenv vilgod -p python3.8
source <home/to/virtualenv>/bin/activate
```

2) Install required packages

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install spconv-cu117
pip install numpy==1.21.5 \
            llvmlite==0.39.0 \
            numba==0.56.4 \
            tensorboardX==2.4.1 \
            easydict==1.9 \
            pyyaml==6.0 \
            scikit-image==0.20.0 \
            tqdm==4.64.0 \
            SharedArray==3.1.0 \
            protobuf==3.19.6 \
            open3d==0.15.2 \
            gpustat==1.0.0 \
            av2==0.2.0 \
            kornia==0.5.8 \
            waymo-open-dataset-tf-2-11-0

pip install hdbscan \
            hydra-core \
            ftfy \
            regex \
            pyransac3d \
            fvcore \
            torch_scatter \
            filterpy

pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt1131/download.html

pip install numpy==1.23.5
```

### Clone and install required repositories
1) Clone repository and create folder structure
```bash
git clone git@github.com:chreisinger/ViLGOD.git
cd ViLGOD
mkdir models
mkdir data
cd models
mkdir clip
cd ..
python setup.py develop
```

2) Insall adapted Patchwork++

```bash
cd third_party/patchwork-plusplus
python setup.py install
```

3) Download [clip model](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt) to: ViLGOD/models/clip

4) Install OpenPCDet (outside of ViLGOD folder)
```bash
git clone https://github.com/open-mmlab/OpenPCDet.git
cd OpenPCDet
python setup.py develop
```
5) Extract data following the [OpenPCDet tutorial](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md). No ground truth database needed!

6) Create softlinks of your extracted data into ViLGOD (we support [Waymo Open Dataset v1.2](https://waymo.com/open/) and [Argoverse 2](https://www.argoverse.org/av2.html))

```bash
ln -s <path/to/extracted/data> ViLGOD/data/
```

## Run ViLGOD
### Run unsupervised 3D object detection

Make sure the CLIP folder is part of your python path:
```
export PYTHONPATH=${PYTHONPATH}:<path/to/ViLGOD>/third_party/CLIP
```
For the Waymo Open Dataset:
```bash
cd tools
python preprocess_data.py preprocessor=waymo
```

For the Argoverse 2 dataset:
```bash
cd tools
python preprocess_data.py preprocessor=argoverse
```

## 📖 Citation

If you find our code or paper helpful, please leave a ⭐ and cite us:

```bibtex
@inproceedings{fruhwirth2024vilgod,
    title={Vision-Language Guidance for LiDAR-based Unsupervised 3D Object Detection}, 
    author={Christian Fruhwirth-Reisinger and Wei Lin and Dušan Malić and Horst Bischof and Horst Possegger},
    year={2024},
    booktitle={British Machine Vision Conference}
}
```

## 🙌 Acknowledgments
Many thanks to [Patchwork++](https://github.com/url-kaist/patchwork-plusplus), [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), [MODEST](https://github.com/YurongYou/MODEST?tab=readme-ov-file), [CLIP](https://github.com/openai/CLIP) and [PointCLIPv2](https://github.com/yangyangyang127/PointCLIP_V2) for code and models.
