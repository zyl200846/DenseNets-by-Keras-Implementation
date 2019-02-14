**Keras implementation of DenseNets**

Original paper: https://arxiv.org/abs/1608.06993
Original implementation: https://github.com/liuzhuang13/DenseNet

@inproceedings{huang2017densely,
  title={Densely connected convolutional networks},
  author={Huang, Gao and Liu, Zhuang and van der Maaten, Laurens and Weinberger, Kilian Q },
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2017}
}

Introduction to each folder and file:
"Data": put the CIFAR-100 data set here
"log": training log for model with no augmentation, relates to main.py
"Pictures": the place store generated visualized samples, relates to get_img_samples_and_conv_results.py
"Preprocess": the folder stores pre-processing files
              ** load_data.py: load CIFAR-100 data set
              ** normalize.py: functions used to normalize data by mean and variance way
              ** utils: some functions used to visualize samples from data set
"weights": stores trained model weights with no augmentation


main.py: run this Python script if you want to test the data with no augmentation
main_aug.py: run this Python script if you want to evaluate data with augmentation
model.py: model building using Keras
predict.py: Run this Python script to see predicted results and generate confusion matrix, change file names
            or file paths to get corresponding data for successful running



Notice: this is the re-implementation of DenseNets, but not detailed ones as same as the original paper.
        Differences like pre-processing, normalize method and hyper-parameters settings.



If you have any problem, please contact Jielong ZHONG with email: jielong26@outlook.com
