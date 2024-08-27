
# MESM

The official code of [Towards Balanced Alignment: Modal-Enhanced Semantic Modeling for Video Moment Retrieval](https://arxiv.org/abs/2312.12155) (AAAI 2024).

## Training

You can run `train.py` with arguments in the command line:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py {--args}
```

Or run with a config file as input:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config_file ./config/charades/VGG_GloVe.json
```

## Evaluation

You can run `eval.py` with arguments in the command line:

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py {--args}
```

Or run with a config file as input:

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --config_file ./config/charades/VGG_GloVe_eval.json
```

For more details, you can refer to the MESM repository and the provided documentation.
