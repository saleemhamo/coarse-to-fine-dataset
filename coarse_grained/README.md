
# Coarse-Grained Text-Video Retrieval using T-MASS

This component of the coarse-to-fine-grained text-video retrieval pipeline utilizes the T-MASS model, which was originally developed for efficient text-video retrieval by modeling text as stochastic embeddings. The approach enriches text embeddings with a flexible and resilient semantic range, improving retrieval accuracy.

## Original Work Reference

The T-MASS model was introduced in the paper:  
**"Text Is MASS: Modeling as Stochastic Embedding for Text-Video Retrieval"**  
Authors: [Jiamian Wang](https://jiamian-wang.github.io/), [Guohao Sun](https://scholar.google.com/citations?user=tf2GWowAAAAJ&hl=en), [Pichao Wang](https://wangpichao.github.io/), [Dongfang Liu](https://dongfang-liu.github.io/), [Sohail Dianat](https://www.rit.edu/directory/sadeee-sohail-dianat), [Majid Rabbani](https://www.rit.edu/directory/mxreee-majid-rabbani), [Raghuveer Rao](https://ieeexplore.ieee.org/author/37281258600), [Zhiqiang Tao](https://ztao.cc/).

- [Paper](https://arxiv.org/abs/2403.17998)  
- [Supplementary Material](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Wang_Text_Is_MASS_CVPR_2024_supplemental.pdf)  
- [Poster](https://drive.google.com/file/d/1HNQ9kDYeegRWG_GuXzTubbPCPjEuDPlA/view?usp=sharing)  
- [Pretrained Models](https://drive.google.com/drive/folders/165PUfnutKRj2_cgjHZE3mIIa8f4kmY6x?usp=sharing)


To run the training and testing for the T-MASS approach, use the following commands:

### Training:
```
python train.py --arch=clip_stochastic --exp_name=MSR-VTT-9k --videos_dir={VIDEO_DIR} --batch_size=32 --noclip_lr=3e-5 --transformer_dropout=0.3 --dataset_name=MSRVTT --msrvtt_train_file=9k --stochasic_trials=20 --gpu='0' --num_epochs=5 --support_loss_weight=0.8
```
### Evaluation:
```
python test.py --datetime={FOLDER_NAME_UNDER_MSR-VTT-9k} --arch=clip_stochastic --videos_dir={VIDEO_DIR} --batch_size=32 --noclip_lr=3e-5 --transformer_dropout=0.3 --dataset_name=MSRVTT --msrvtt_train_file=9k --stochasic_trials=20 --gpu='0' --load_epoch=0 --exp_name=MSR-VTT-9k
```

