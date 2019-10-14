# STELA: A Real-Time Scene Text Detector with Learned Anchor


STELA is a simple and intuitive method for multi-oriented text detection based on RetinaNet. The key idea is utilizing the learned anchor which is obtained through a regression operation to replace the original into the final predictions. In our experiments, it achieves an F-measure 0.887 on ICDAR 2013, 0.833 on ICDAR 2015 and 0.715 on ICDAR 2017 MLT. For more details, please refer to our paper. 


This code is modified from [RetinaNet](https://github.com/yhenon/pytorch-retinanet) and [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). It has been tested on Ubuntu 16.04 with CUDA 9.0 and [PyTorch](https://github.com/pytorch/pytorch) 1.1. We think this code is easy to use, so we omitted a lot of instructions. If you have some issues, please leave a message.


At last, if you find the paper and code useful in your research, please consider citing:

@article{deng2019stela,
    Title = {STELA: A Real-Time Scene Text Detector with Learned Anchor},
    Author = {Linjie Deng and Yanxiang Gong and Xinchen Lu and Yi Lin and Zheng Ma and Mei Xie},
    Journal = {arXiv preprint arXiv:1909.07549},
    Year = {2019}
}
















