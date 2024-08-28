# Modal-Enhanced Semantic Modeling for Video Moment Retrieval (MESM)

This project component implements the MESM model, which is part of a broader coarse-to-fine-grained text-video retrieval pipeline. MESM addresses the challenge of modality imbalance in video moment retrieval (VMR) by enhancing the alignment between video and text modalities, ensuring balanced alignment at both the frame-word and segment-sentence levels.

## Original Work Reference

The MESM model was introduced in the paper:  
**"Towards Balanced Alignment: Modal-Enhanced Semantic Modeling for Video Moment Retrieval"**  
Authors: [Zhihang Liu](https://github.com/lntzm), [Jun Li](#), [Hongtao Xie](#), [Pandeng Li](#), [Jiannan Ge](#), [Sun-Ao Liu](#), [Guoqing Jin](#).

- [Paper](https://arxiv.org/abs/2312.12155)  
- [Supplementary Material](#)  
- [Poster](#)  
- [Pretrained Models](#)

## Training

Run `train.py` with command-line arguments:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py {--args}
```

Or run it with a configuration file:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config_file ./config/charades/VGG_GloVe.json
```

## Evaluation

Run `eval.py` with command-line arguments:
```bash
CUDA_VISIBLE_DEVICES=0 python eval.py {--args}
```

Or run it with a configuration file:
```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --config_file ./config/charades/VGG_GloVe_eval.json
```
