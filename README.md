# FArMARe
This repository contains the source code for the paper ["Furniture-Aware Multi-task methodology for Recommending Apartments based on the user interests"](https://openaccess.thecvf.com/content/ICCV2023W/CV4Metaverse/html/Abdari_FArMARe_a_Furniture-Aware_Multi-Task_Methodology_for_Recommending_Apartments_Based_on_ICCVW_2023_paper.html).

![ICCVW23_ApartmentsRanking (1)_page-0001](https://github.com/aliabdari/FArMARe/assets/24971267/54c56004-6680-4755-841e-3f5b316c51d2)

This paper has been accepted in the ICCV 2023, Computer Vision for Metaverse (CV4Metaverse) Workshop. 

## Metaverse Apartment Recommendation Challenge
Also, this repository can be used as the base code for the [Metaverse Apartment Recommendation Challenge](http://ailab.uniud.it/apartment-recommendation-challenge).

## Data
This work is based on the [3DFRONT](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset) dataset with the corresponding descriptions for each scenario. The 3DFRONT dataset can be obtained [here](https://tianchi.aliyun.com/dataset/65347). The descriptions can be found in the [descriptions](https://github.com/aliabdari/FArMARe/tree/main/descriptions) directory. On the other side, the point-of-view images for each scene have been obtained from [this repository](https://github.com/xheon/panoptic-reconstruction) which could be downloaded through [front3d-2d.zip](https://kaldir.vc.in.tum.de/panoptic_reconstruction/front3d-2d.zip).

You can run the following commands inside the project root directory to download the point-of-view images.
```
wget https://kaldir.vc.in.tum.de/panoptic_reconstruction/front3d-2d.zip 
unzip front3d-2d.zip -d front3d_2d_images
```
## Features
To obtain the features you can execute scripts existing in the [features_generation](https://github.com/aliabdari/FArMARe/tree/main/features_generation) to obtain either Resnet/Bert features or OpenClip features for both point-of-view images and the descriptions. Also, the prepared features can be obtained through [open_clip_features](https://drive.google.com/file/d/1GfB1UHSb1KCqKLFjpk80I3Oi08ocwIBi/view?usp=sharing) and [resnet_bert_features](https://drive.google.com/file/d/1jDT7MUl3VWJY7fs6GlcWYUsI041V98Tm/view?usp=sharing). After downloading put them in the root directory of the project.

## Environment
In this work, Python3.10 and PyTorch 1.13.1 have been used.

## Train and Evaluation
The module of training different models explained in the paper can be found in the [train_evaluiation](https://github.com/aliabdari/FArMARe/tree/main/train_evaluation) directory.

- open_clip_one_dimensional.py: To train using the GRU/BIGRU model for the textual part, use a 1d convolutional for the imagery part(Using open clip features)
- open_clip_containing_classifier.py: To train using the GRU/BIGRU model for the textual part, use a 1d convolutional for the imagery part beside a classifier(Using open clip features)
- open_clip_mean.py: To train using the GRU/BIGRU model for the textual part, use a fully connected network for the imagery part(Using open clip features)
- resnet_bert_one_dimensional.py: To train using the GRU/BIGRU model for the textual part, use a 1d convolutional for the imagery part(Using Bert, Resnet features for the textual and imagery parts, respectively)
- resnet_bert_containing_classifier.py: To train using the GRU/BIGRU model for the textual part, use a 1d convolutional for the imagery part beside a classifier(Using Bert, Resnet features for the textual and imagery parts, respectively)
- resnet_bert_mean.py: To train using the GRU/BIGRU model for the textual part, use a fully connected network for the imagery part(Using Bert, Resnet features for the textual and imagery parts, respectively)

To start training and getting the evaluation on the test set (for example) you can run the following:
```
python open_clip_one_dimensional.py
```
The models will be saved in train_evaluation/models path.
There are several arguments which could be set to train models, explained in the following:
### Arguments

- --output_feature_size: It specifies the output feature size of the visual model and the text model/ default=256
- --is_bidirectional: Since in some of the modules GRU has been used to process textual descriptions, this argument allows the use of GRU or BiGRU models / default=True
- --is_token_level: Since in this work we have used sentence-level features and token-level features, this argument allows us to switch between these two modes / default=False
- --num_epochs: Specifies the number of epochs/ default=50
- --batch_size: Specifies the batch size/ default=64
- --lr: Specifies the learning rate/ default=0.008
- --step_size: Since in this work the decay technique of the learning rate has been used this argument specifies after how many epochs the decay occurs / default=27
- --gamma: Accordingly, this argument specifies which percentage the decay in learning rate occurs / default=0.75

## Citation
If you found this repository or paper useful please cite this work.

```
@inproceedings{abdari2023farmare,
  title={FArMARe: a Furniture-Aware Multi-task methodology for Recommending Apartments based on the user interests},
  author={Abdari, Ali and Falcon, Alex and Serra, Giuseppe},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4293--4303},
  year={2023}
}
```

