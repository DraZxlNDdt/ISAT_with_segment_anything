pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.1 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.28.2
pip install transformers

DOWNLOAD sam_vit_h_4b8939.pth to segment_any
DOWNLOAD vitae_rvsa_kvdiff.pth to rvsa/checkpoints