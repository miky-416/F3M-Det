# F3M-Det
Implementation for the paper "Exploiting Cross-Modal Feature Enhancement, Alignment, and Fusion for Misaligned Visible-Infrared Object Detection"

## Usage
### Install

- Create a conda virtual environment and activate it:

```bash
conda create -n f3m python=3.7
conda activate f3m
```

- Install `PyTorch` and `torchvision` :

Install torch==1.10.1 with CUDA==11.1:

```bash
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

- Install deformable convolution

```bash
git clone https://github.com/miky-416/F3M-Det
cd models/Deformable_Convolution_V2/
pip sh make.sh
cd ../..
```

- Install other requirements:
```bash
pip install -r requirements.txt
```

### Data Preparation

Download datasets ([DVTOD](https://github.com/VDT-2048/DVTOD), [LLVIP](https://github.com/bupt-ai-cz/LLVIP)), and move (or link) them to `F3M-Det/datasets`.

### Evaluation

To evaluate our `F3M-Det` on DVTOD, run:

```bash
python test.py <config-file> <checkpoint>
```

**Note**: You can download checkpoint files from [google drive](https://drive.google.com/drive/folders/1U63VX8pZrV--8ks_VHXeB37Kgf447vN8?usp=drive_link).

