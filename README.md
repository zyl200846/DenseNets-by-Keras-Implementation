**Keras implementation of DenseNets**

Original paper: https://arxiv.org/abs/1608.06993 <br>Original implementation: https://github.com/liuzhuang13/DenseNet</br>

@inproceedings{
  <br>&emsp;&emsp;huang2017densely,</br>
  &emsp;&emsp;title={Densely connected convolutional networks},</br>
  &emsp;&emsp;author={Huang, Gao and Liu, Zhuang and van der Maaten, Laurens and Weinberger, Kilian Q },</br>
  &emsp;&emsp;booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},</br>
  &emsp;&emsp;year={2017}</br>
}

**Introduction to each folder and file:**</br>
&emsp; "Data": put the CIFAR-100 data set here</br>
&emsp; "log": training log for model with no augmentation, relates to main.py</br>
&emsp; "Pictures": the place store generated visualized samples, relates to get_img_samples_and_conv_results.py</br>
&emsp; "Preprocess": the folder stores pre-processing files</br>
              &emsp;&emsp;&emsp; load_data.py: load CIFAR-100 data set</br>
              &emsp;&emsp;&emsp; normalize.py: functions used to normalize data by mean and variance way</br>
              &emsp;&emsp;&emsp; utils: some functions used to visualize samples from data set</br>
&emsp; "weights": stores trained model weights with no augmentation</br></br>

&emsp; main.py: run this Python script if you want to test the data with no augmentation</br>
&emsp; main_aug.py: run this Python script if you want to evaluate data with augmentation</br>
&emsp; model.py: model building using Keras</br>
&emsp; predict.py: Run this Python script to see predicted results and generate confusion matrix, change file names or file paths to get corresponding data for successful running</br></br>

Notice: this is the re-implementation of DenseNets, but not detailed ones as same as the original paper.
        Differences like pre-processing, normalize method and hyper-parameters settings.</br>

If you have any problem, please contact Jielong ZHONG with email: jielong26@outlook.com</br>
