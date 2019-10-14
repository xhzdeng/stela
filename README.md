# STELA: A Real-Time Scene Text Detector with Learned Anchor


STELA is a simple and intuitive method for multi-oriented text detection based on RetinaNet. The key idea is utilizing the learned anchor which is obtained through a regression operation to replace the original into the final predictions. In our experiments, it achieves an F-measure 0.887 on ICDAR 2013, 0.833 on ICDAR 2015 and 0.715 on ICDAR 2017 MLT. For more details, please refer to our [paper](https://arxiv.org/abs/1909.07549). 

### Installation

This code is modified from [RetinaNet](https://github.com/yhenon/pytorch-retinanet) and [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). It has been tested on Ubuntu 16.04 with CUDA 9.0 and [PyTorch](https://github.com/pytorch/pytorch) 1.1. If you have some issues, please leave a message.

0. Clone this repository
    ```
    git clone https://github.com/xhzdeng/stela.git
    ```

1. Build the Cython modules
    ```
	cd $STELA_ROOT/utils
	sh make.sh
    ```

2. Prepare your own data directory. For our implement, it should follow the format of [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/).

3. Train with YOUR dataset
    ```
    cd $STELA_ROOT
    python train.py
    ```

4. Test with YOUR models
    ```
    cd $STELA_ROOT
    python eval.py
    ```


### Citation

At last, if you find the paper and code useful in your research, please consider citing:

	@article{deng2019stela,
		Title = {STELA: A Real-Time Scene Text Detector with Learned Anchor},
		Author = {Linjie Deng and Yanxiang Gong and Xinchen Lu and Yi Lin and Zheng Ma and Mei Xie},
		Journal = {arXiv preprint arXiv:1909.07549},
		Year = {2019}
	}
















