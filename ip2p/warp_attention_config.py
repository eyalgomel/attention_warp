# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Instruct-GS2GS Inject Attention configuration file.
"""

import sys
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.plugins.types import MethodSpecification


from igs2gs.igs2gs_datamanager import InstructGS2GSDataManagerConfig
from ip2p._2dgs import InstructGS2GSModelConfig
from igs2gs.igs2gs_trainer import InstructGS2GSTrainerConfig

from ip2p.warp_attention_pipeline import WarpAttentionPipelineConfig

max_num_iterations = 3000
for s in ["--max_num_iterations", "--max-num-iterations"]:
    try:
        idx = sys.argv.index(s)
        max_num_iterations = int(sys.argv[idx+1])
        break
    except ValueError:
        pass

start_from_iter = 30_000

attn_warp_method = MethodSpecification(
    config=InstructGS2GSTrainerConfig(
        method_name="attn_warp",
        steps_per_eval_image=100,
        steps_per_eval_batch=100,
        steps_per_save=500,
        steps_per_eval_all_images=100000, 
        max_num_iterations=max_num_iterations,
        mixed_precision=False,
        gradient_accumulation_steps = {'color':10,'shs':10},
        pipeline=WarpAttentionPipelineConfig(
            datamanager=InstructGS2GSDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(load_3D_points=True,
                                                      train_split_fraction=1.),
            ),
            model=InstructGS2GSModelConfig(output_depth_during_training=True),
        ),
    optimizers={
        "means": {
            "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1.6e-6,
                max_steps=start_from_iter + max_num_iterations,
                warmup_steps=start_from_iter
            ),
        },
        "features_dc": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "features_rest": {
            "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
            "scheduler": None,
        },
        "opacities": {
            "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
            "scheduler": None,
        },
        "scales": {
            "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
            "scheduler": None,
        },
        "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
    },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Diffusion-Based Attention Warping for Consistent 3D Scene Editing",
)
