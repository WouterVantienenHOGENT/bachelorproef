# P2PNet inference adapter

Source: https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet
Commit: `5c91a81ca062b1c7fd3db3ad1c55b1c21f0a7455`.

`network.py` retains the inference architecture from `models/p2pnet.py`.
`backbone.py` retains the VGG16-BN layer partitioning from `models/backbone.py`.
Training losses, matching and distributed helpers were removed. Anchor tensors
follow the input device instead of choosing CUDA globally. VGG features are
constructed with torchvision without downloading separate ImageNet weights.
All checkpoint parameters, including unused head layers, are retained for strict loading.

The complete upstream license is in LICENSE. P2PNet permits academic research use only.
The weights are downloaded unchanged from the same pinned official repository.
