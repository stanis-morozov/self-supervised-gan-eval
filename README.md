# On Self-Supervised Image Representations For GAN Evaluation

This repository is the official implementation of self-supervised FID from **On Self-Supervised Image Representations For GAN Evaluation** by Stanislav Morozov, Andrey Voynov and Artem Babenko.

FID is a measure of similarity between two datasets of images. 
It was shown to correlate well with human judgement of visual quality and is most often used to evaluate the quality of samples of Generative Adversarial Networks.
Self-supervised FID is calculated by computing the [Fréchet distance](https://en.wikipedia.org/wiki/Fr%C3%A9chet_distance) between two Gaussians fitted to feature representations of the [SwAV](https://github.com/facebookresearch/swav) Resnet50 network.


## Installation

```
python3 setup.py install
```

Requirements:
- python3
- numpy
- pillow
- scipy
- torch>=1.7.0
- torchvision>=0.8.1

## Usage

To compute the self-supervised FID score between two datasets, where images of each dataset are contained in an individual folder:
```
python3 -m self-supervised-gan-eval path/to/dataset1 path/to/dataset2
```
Optionally, you can limit the number of images to calculate the FID in each dataset.
```
python3 -m self-supervised-gan-eval path/to/dataset1 path/to/dataset2 --max-size MAX_SIZE
```

To run the evaluation on GPU, use the flag `--gpu N`, where `N` is the index of the GPU to use. 

## Human evaluation

Human evaluation labels can be downloaded via the following links:
* [Precision](https://www.dropbox.com/s/f785t53c7jgx6he/precision_labels.csv?dl=1)
* [Recall](https://www.dropbox.com/s/u0lugzjq8qmcz58/recall_labels.csv?dl=1)
* [TopK](https://www.dropbox.com/s/rg45jcrnna5s9c7/topk_labels.csv?dl=1)

Precision and Recall tables have the following columns:
* ```image_ref```, ```image_left```, ```image_right``` contain paths to the reference, left, and right images, respectively
* ```embedding``` corresponds to the embedding that was used to calculate the nearest neighbors
* ```left_score```, ```right_score```, ```equal_good_score```, ```equal_bad_score``` correspond to how the assessors voted. Each pair was labeled by 9 independent assessors

TopK table has the following columns:
* ```image_ref```, ```images_left```, ```images_right``` contain paths to the reference image and blocks of left and right images, respectively. There is also possibility to get individual images. For example, if the path to the image block is ```topk_data/CelebaHQ/Resnet50_SWAV/2rows_3cols/18099_20532_5497_2490_15318.jpg```, then it contains images:
  * ```CelebaHQ/real/18099.jpg```
  * ```CelebaHQ/real/20532.jpg```
  * ```CelebaHQ/real/5497.jpg```
  * ```CelebaHQ/real/2490.jpg```
  * ```CelebaHQ/real/15318.jpg```
* ```embedding_left```, ```embedding_right``` correspond to the embeddings that were used to calculate the top-k for the left and right blocks
* ```left_score```, ```right_score```, ```equal_score``` correspond to how the assessors voted. Each task was labeled by 9 independent assessors

We also release the dataset with the images used for labeling and embedding vectors (InceptionV3 and SwAV) for all images in the dataset:
* Individual image dataset can be downloaded via ```download.sh```. **Warning!** The dataset requires at least 160 GB of disk space
* Dataset with top-5 blocks for the top-k task can be downloaded via [the following link](https://www.dropbox.com/s/5euz181r5mui4bl/topk_data.tar.gz?dl=1)
* [Embeddings](https://www.dropbox.com/s/cqtars98ps64jgs/embeddings.tar.gz?dl=1)

## Citing

If you use this repository in your research, consider citing it using the following paper:

```
@inproceedings{morozov2021on,
  title={On Self-Supervised Image Representations for GAN Evaluation},
  author={Morozov, Stanislav and Voynov, Andrey and Babenko, Artem},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```
