# Replication Instruction

## Installation

1. Follow the requirement installation for [BLIP repository]([GitHub - salesforce/BLIP: PyTorch code for BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://github.com/salesforce/BLIP/)).

2. Download checkpoints and put into `ckpts` path.

3. Paste code in the repository to the cloned BLIP path.

4. Follow [ScanQA repository]([GitHub - ATR-DBI/ScanQA](https://github.com/ATR-DBI/ScanQA/)). Download and prepeocess data. 

5. Replace the ScanQA data path in the code to yours.

## Scene Views Generation

1. Replace the Scannet data path in the `render_scenes.py` to yours.

2. Run  `render_scenes.py`.

## Zero-Shot Eval

1. Run `eval_scene_best_views.py` to zero-shot evaluate BLIP with ScanQA.

2. A result json will be generated, indicating matched views w.r.t. questions.

## Train BLIP with Views

1. Run  `train_scene_view_vqa.py`.

