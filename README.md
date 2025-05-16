# Diffusion-Based Attention Warping for Consistent 3D Scene Editing

This repository contains the implementation of our method for advanced 3D scene editing. Please follow the setup instructions to ensure proper installation and usage. 
Our method leverages diffusion models' attention mechanisms to enable precise, controllable 3D scene editing while preserving global consistency and structure across multiple viewpoints.


[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://arxiv.org/abs/2412.07984)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://attention-warp.github.io)


<img src="images/variations.png" alt="Overview of our Diffusion-Based Attention Warping method" width="600">


## Installation

Follow the instructions provided by [NeRFStudio](https://github.com/nerfstudio-project/nerfstudio) to set up the environment. Below is an example of how to create and configure the environment for this repository. 

### Example Environment Setup

```bash
# Create and activate a Conda environment
conda create --name attn_warp -y python=3.8
conda activate attn_warp
python -m pip install --upgrade pip

# Install PyTorch and CUDA dependencies according to NeRFStudio
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

# Install NeRFStudio and other dependencies
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install nerfstudio==1.1.4
pip install git+https://github.com/nerfstudio-project/gsplat.git@ec3e715f5733df90d804843c7246e725582df10c
pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git@05c386ee95b26a8ec8398bebddf70ffb8ddd3faf
pip install -e .
pip install git+https://github.com/cvachha/instruct-gs2gs

# Workaround
ATTN_WARP_PATH=$(conda env list | grep attn_warp | awk '{print $NF}')
pip install git+https://github.com/nerfstudio-project/gsplat.git@ec3e715f5733df90d804843c7246e725582df10c
pip install --no-deps seaborn
pip install tyro==0.6.6
sed -i '/def compute(self) -> Tensor:/,/return _lpips_compute/c\    def compute(self) -> Tensor:\n        self.loss = _lpips_compute(self.sum_scores, self.total, self.reduction);return self.loss' "$ATTN_WARP_PATH/lib/python3.8/site-packages/torchmetrics/image/lpip.py"
sed -i 's/^from gsplat\.cuda_legacy\._wrapper import num_sh_bases/#&/' "$ATTN_WARP_PATH/lib/python3.8/site-packages/nerfstudio/models/splatfacto.py"

# Install the NeRFStudio CLI
ns-install-cli
```

## Running the Method

### Step 1: Train a 2DGS Scene
You need to first train a 2DGS scene using the following command:
```bash
ns-train splatfacto-2dgs --output-dir OUTPUT_DIR --experiment-name EXP_NAME --data DATA_PATH
```

### Step 2: Run the Method
To run the method, use the following command:
```bash
ns-train attn_warp --load-dir 2DGS_MODEL/nerfstudio_models --pipeline.datamanager.data DATA_PATH \
--pipeline.model_type ip2p \
--experiment-name EXP_NAME \
--output-dir OUTPUT_DIR \
--pipeline.prompt EDIT_PROMPT \
--pipeline.edit_idx EDIT_INDEX
```

### Optional Flags
- **Masking with LangSAM**:  
  Add the following flag to use LangSAM for masking:
  ```bash
  --pipeline.mask_prompt MASK_PROMPT
  ```

- **Evaluation**:  
  Use the following flags to enable evaluation:
  ```bash
  --pipeline.evaluate True \
  --pipeline.src_prompt SRC_PROMPT \
  --pipeline.target_prompt TARGET_PROMPT \
  --pipeline.novel_view_path CAMERA_NOVEL_VIEW_PATH
  ```

### Full Parameter List
For the complete list of parameters, refer to the file: [ip2p/warp_attention_pipeline.py](ip2p/warp_attention_pipeline.py#L709)

## Acknowledgments
This project is built upon [NeRFStudio](https://github.com/nerfstudio-project/nerfstudio) and utilizes components from:
- [Instruct-GS2GS](https://github.com/cvachha/instruct-gs2gs)
- [DDPM-Inversion](https://github.com/inbarhub/DDPM_inversion)
- [InstructNeRF2NeRF](https://github.com/ayaanzhaque/instruct-nerf2nerf)


## Citation
If you find this work useful, please consider citing our paper:

> ```bibtex
> @misc{gomel2024diffusionbasedattentionwarpingconsistent,
>    title={Diffusion-Based Attention Warping for Consistent 3D Scene Editing}, 
>    author={Eyal Gomel and Lior Wolf},
>    year={2024},
>    eprint={2412.07984},
>    archivePrefix={arXiv},
>    primaryClass={cs.CV},
>    url={https://arxiv.org/abs/2412.07984}, 
> }
> ```
