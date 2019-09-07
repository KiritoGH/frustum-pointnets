## 代码链接：https://github.com/charlesq34/frustum-pointnets
## 参考：https://github.com/chonepieceyb/reading-frustum-pointnets-code

## Frustum PointNets for 3D Object Detection from RGB-D Data
Created by <a href="http://charlesrqi.com" target="_blank">Charles R. Qi</a>, <a href="http://www.cs.unc.edu/~wliu/" target="_black">Wei Liu</a>, <a href="http://www.cs.cornell.edu/~chenxiawu/" target="_blank">Chenxia Wu</a>, <a href="http://cseweb.ucsd.edu/~haosu/" target="_blank">Hao Su</a> and <a href="http://geometry.stanford.edu/member/guibas/" target="_blank">Leonidas J. Guibas</a> from <a href="http://www.stanford.edu" target="_blank">Stanford University</a> and <a href="http://nuro.ai" target="_blank">Nuro Inc.</a>

![teaser](https://github.com/charlesq34/frustum-pointnets/blob/master/doc/teaser.jpg)

## Introduction
This repository is code release for our CVPR 2018 paper (arXiv report [here](https://arxiv.org/abs/1711.08488)). In this work, we study 3D object detection from RGB-D data. We propose a novel detection pipeline that combines both mature 2D object detectors and the state-of-the-art 3D deep learning techniques. In our pipeline, we firstly build object proposals with a 2D detector running on RGB images, where each 2D bounding box defines a 3D frustum region. Then based on 3D point clouds in those frustum regions, we achieve 3D instance segmentation and amodal 3D bounding box estimation, using PointNet/PointNet++ networks (see references at bottom).

此仓库是CVPR 2018论文的代码版本。本文研究了基于RGB-D数据的三维物体检测方法。我们提出了一种结合成熟的二维物体检测器和最先进的三维深度学习技术的新颖的检测方案。在我们的方案中，我们首先在RGB图像上运行二维检测器以生成物体提案，每个二维边界框都对应一个三维的截锥体区域。然后，基于这些截锥体区域中的三维点云，我们使用PointNet/PointNet++网络获得了三维实例分割和非模态的三维边界框估计。

By leveraging 2D object detectors, we greatly reduce 3D search space for object localization. The high resolution and rich texture information in images also enable high recalls for smaller objects like pedestrians or cyclists that are harder to localize by point clouds only. By adopting PointNet architectures, we are able to directly work on 3D point clouds, without the necessity to voxelize them to grids or to project them to image planes. Since we directly work on point clouds, we are able to fully respect and exploit the 3D geometry -- one example is the series of coordinate normalizations we apply, which help canocalizes the learning problem. Evaluated on KITTI and SUNRGBD benchmarks, our system significantly outperforms previous state of the art and is still in leading positions on current <a href="http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d">KITTI leaderboard</a>.

通过利用二维物体检测器，我们大大减少了用于物体定位的三维搜索空间。图像的高分辨率和丰富的纹理信息也使得像行人或骑行者这样的小物体具有较高的召回率，这原本是很难仅由点云定位的。通过利用PointNet架构，我们能够直接处理三维点云，而无需将它们体素化或者投影到图像平面。因为我们直接处理点云，所以我们能够充分尊重和利用三维几何信息——一个例子是我们采用一系列坐标标准化，这有助于解决学习问题。根据在KITTI和SUNRGBD基准集上的评估，我们的系统表现明显优于之前的技术水平，并且目前仍处于领先地位。

For more details of our architecture, please refer to our paper or <a href="http://stanford.edu/~rqi/frustum-pointnets" target="_blank">project website</a>.

有关系统架构的更多细节，请查阅论文或项目网址。

## Citation
If you find our work useful in your research, please consider citing:

        @article{qi2017frustum,
          title={Frustum PointNets for 3D Object Detection from RGB-D Data},
          author={Qi, Charles R and Liu, Wei and Wu, Chenxia and Su, Hao and Guibas, Leonidas J},
          journal={arXiv preprint arXiv:1711.08488},
          year={2017}
        }

## Installation
Install <a href="https://www.tensorflow.org/install/">TensorFlow</a>.There are also some dependencies for a few Python libraries for data processing and visualizations like `cv2`, `mayavi`  etc. It's highly recommended that you have access to GPUs.

安装TensorFlow和一些用于数据处理和可视化的Python库的依赖，如`cv2`、`mayavi`等。强烈推荐使用GPU。

To use the Frustum PointNets v2 model, we need access to a few custom Tensorflow operators from PointNet++. The TF operators are included under `models/tf_ops`, you need to compile them (check `tf_xxx_compile.sh` under each ops subfolder) first. Update `nvcc` and `python` path if necessary. The compile script is written for TF1.4. There is also an option for TF1.2 in the script. If you are using earlier version it's possible that you need to remove the `-D_GLIBCXX_USE_CXX11_ABI=0` flag in g++ command in order to compile correctly.

为了使用Frustum PointNets v2模型，我们需要一些来自于PointNet++的自定义TensorFlow操作符。TF操作符包含在`models/tf_ops`目录下，你需要先编译它们（请检查每个ops子目录下的`tf_xxx_compile.sh`）。有必要的话更新`nvcc`和`python`路径。编译脚本是在TF1.4下编写的，另外还有一个TF1.2的脚本。如果你正在使用更早的版本，你可能需要在g++命令中移除`-D_GLIBCXX_USE_CXX11_ABI=0`标志，以确保正确编译。

If we want to evaluate 3D object detection AP (average precision), we need also to compile the evaluation code (by running `compile.sh` under `train/kitti_eval`). Check `train/kitti_eval/README.md` for details.

如果我们想要评估三维物体检测的平均精度，我们还需要编译评估代码（运行`train/kitti_eval`下的`compile.sh`）。查看`train/kitti_eval/README.md`可以获得更多细节。

Some of the demos require `mayavi` library. We have provided a convenient script to install `mayavi` package in Python, a handy package for 3D point cloud visualization. You can check it at `mayavi/mayavi_install.sh`. If the installation succeeds, you should be able to run `mayavi/test_drawline.py` as a simple demo. Note: the library works for local machines and seems do not support remote access with `ssh` or `ssh -X`.

一些演示需要`mayavi`库。我们已经提供了一个方便的脚本用以在Python中安装`mayavi`包，这是一个用作三维点云可视化的便利包。你可以查阅`mayavi/mayavi_install.sh`。如果安装成功，你应该能够运行`mayavi/test_drawline.py`作为一个简单的示例。注意：该库适用于本地机器，不支持使用`ssh`或`ssh -X`的远程访问。

The code is tested under TF1.2 and TF1.4 (GPU version) and Python 2.7 (version 3 should also work) on Ubuntu 14.04 and Ubuntu 16.04 with NVIDIA GTX 1080 GPU. It is highly recommended to have GPUs on your machine and it is required to have at least 8GB available CPU memory.

代码在TF1.2和TF1.4（GPU版本）以及Python2.7（版本3应该也行）下测试，使用NVIDIA GTX 1080 GPU，Ubuntu 14.04和Ubuntu 16.04。强烈推荐使用GPU，并要求至少有8GB的CPU内存。

## Usage

Currently, we support training and testing of the Frustum PointNets models as well as evaluating 3D object detection results based on precomputed 2D detector outputs (under `kitti/rgb_detections`). You are welcomed to extend the code base to support your own 2D detectors or feed your own data for network training.

目前，我们支持Frustum PointNets模型的训练、测试和基于预计算的二维检测器输出（在`kitti/rgb_detections`下）的三维物体检测结果评估。欢迎你拓展代码库以支持你自己的二维检测器或者向网络训练提供自己的数据。

### Prepare Training Data
In this step we convert original KITTI data to organized formats for training our Frustum PointNets. <b>NEW:</b> You can also directly download the prepared data files <a href="https://shapenet.cs.stanford.edu/media/frustum_data.zip" target="_blank">HERE (960MB)</a> -- to support training and evaluation, just unzip the file and move the `*.pickle` files to the `kitti` folder.

在这一步中，我们将原始的KITTI数据转化为有组织的形式以便训练我们的Frustum PointNets。你也可以直接下载准备好的数据文件——为了支持训练和评估，只需解压文件，并把`*.pickle`文件移动到`kitti`目录下。

Firstly, you need to download the <a href="http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d" target="_blank">KITTI 3D object detection dataset</a>, including left color images, Velodyne point clouds, camera calibration matrices, and training labels. Make sure the KITTI data is organized as required in `dataset/README.md`. You can run `python kitti/kitti_object.py` to see whether data is downloaded and stored properly. If everything is fine, you should see image and 3D point cloud visualizations of the data. 

首先，你需要下载KITTI三维物体检测数据集，包含左侧彩色图像，激光点云，相机校准矩阵和训练标签。确保KITTI数据按照`dataset/README.md`的要求进行组织。你可以运行`python kitti/kitti_object.py`来查看数据是否被正确下载和存储。如果一切正常，你应该能看见图像和三维点云的可视化。

然后，为了准备数据，只需运行sh scripts/command_prep_data.sh（警告：这一步会生成大约4.7GB的pickle文件数据）

Then to prepare the data, simply run: (warning: this step will generate around 4.7GB data as pickle files)

    sh scripts/command_prep_data.sh

Basically, during this process, we are extracting frustum point clouds along with ground truth labels from the original KITTI data, based on both ground truth 2D bounding boxes and boxes from a 2D object detector. We will do the extraction for the train (`kitti/image_sets/train.txt`) and validation set (`kitti/image_sets/val.txt`) using ground truth 2D boxes, and also extract data from validation set with predicted 2D boxes (`kitti/rgb_detections/rgb_detection_val.txt`).

基本上，在这个过程中，基于二维边界框真值和二维物体检测器结果，我们从原始KITTI数据提取截锥体点云和真值标签。我们将使用二维边框真值来提取训练（`kitti/image_sets/train.txt`）和测试集（`kitti/image_sets/val.txt`），并使用预测的二维边框（`kitti/rgb_detections/rgb_detection_val.txt`）从验证集中提取数据。

You can check `kitti/prepare_data.py` for more details, and run `python kitti/prepare_data.py --demo` to visualize the steps in data preparation.

你可以查阅`kitti/prepare_data.py`来获得更多细节，并运行`python kitti/prepare_data.py --demo`来可视化数据准备步骤。

After the command executes, you should see three newly generated data files under the `kitti` folder. You can run `python train/provider.py` to visualize the training data (frustum point clouds and 3D bounding box labels, in rect camera coordinate).

执行这条命令后，你应该可以看到在`kitti`目录下有三个新生成的数据文件。你可以运行`python train/provider.py`来可视化训练数据（在矩形相机坐标系下的截锥体点云和三维边界框标签）

### Training Frustum PointNets

To start training (on GPU 0) the Frustum PointNets model, just run the following script:

    CUDA_VISIBLE_DEVICES=0 sh scripts/command_train_v1.sh

You can run `scripts/command_train_v2.sh` to trian the v2 model as well. The training statiscs and checkpoints will be stored at `train/log_v1` (or `train/log_v2` if it is a v2 model). Run `python train/train.py -h` to see more options of training. 

为了在GPU 0开始训练Frustum PointNets模型，只需运行下面的脚本：CUDA_VISIBLE_DEVICES=0 sh scripts/command_train_v1.sh

你也可以运行`scripts/command_train_v2.sh`来训练v2模型。训练统计和检查点将被存储在`train/log_v1`（或`train/log_v2`，如果是v2模型）。运行`python train/train.py -h`来查看更多的训练选项。

<b>NEW:</b> We have also prepared some pretrained snapshots for both the v1 and v2 models. You can find them <a href="https://shapenet.cs.stanford.edu/media/frustum_pointnets_snapshots.zip" target="_blank">HERE (40MB)</a> -- to support evaluation script, you just need to unzip the file and move the `log_*` folders to the `train` folder.

我们也已经为v1和v2模型准备了一些预训练的快照。你可以找到它们——为了支持评估脚本，你也需要解压文件并移动`log_*`目录到`train`目录。

### Evaluation
To evaluate a trained model (assuming you already finished the previous training step) on the validation set, just run:

    CUDA_VISIBLE_DEVICES=0 sh scripts/command_test_v1.sh

Similarly, you can run `scripts/command_test_v2.sh` to evaluate a trained v2 model. The script will automatically evaluate the Frustum PointNets on the validation set based on precomputed 2D bounding boxes from a 2D detector (not released here), and then run the KITTI offline evaluation scripts to compute precision recall and calcuate average precisions for 2D detection, bird's eye view detection and 3D detection.

为了在验证集上评估训练好的模型（假定你已经完成了先前的训练步骤），只需运行：CUDA_VISIBLE_DEVICES=0 sh scripts/command_test_v1.sh

类似地，你可以运行`scripts/command_test_v2.sh`来评估训练好的v2模型。这个脚本会基于从二维检测器（这里没有公布）预计算的二维边界框，在验证集上自动评估Frustum PointNets，然后运行KITTI离线评估脚本来计算精确率、召回率以及二维检测、鸟瞰检测和三维检测的平均精度。

Currently there is no script for evaluation on test set, yet it is possible to do it by yourself. To evaluate on the test set, you need to get outputs from a 2D detector on KITTI test set, store it as something in `kitti/rgb_detections`. Then, you need to prepare test set frustum point clouds for the test set, by modifying the code in `kitti/prepare_data.py`. Then you can modify test scripts in `scripts` by changing the data path, idx path and output file name. For our test set results reported, we used the entire `trainval` set for training.

目前还没有用于测试集评估的脚本，但是你可以自己编写。为了在测试集上进行评估，你需要获得KITTI测试集的二维检测器输出，将其存储在`kitti/rgb_detections`。然后，你需要修改`kitti/prepare_data.py`的代码，准备测试集截锥体点云。接着，你可以修改`scripts`下的测试脚本，改变数据路径、索引路径和输出文件名。对于我们公布的测试集结果，我们使用了整个`trainval`集进行训练。

## License
Our code is released under the Apache 2.0 license (see LICENSE file for details).

## References
* <a href="http://stanford.edu/~rqi/pointnet" target="_blank">PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation</a> by Qi et al. (CVPR 2017 Oral Presentation). Code and data: <a href="https://github.com/charlesq34/pointnet">here</a>.
* <a href="http://stanford.edu/~rqi/pointnet2" target="_black">PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space</a> by Qi et al. (NIPS 2017). Code and data: <a href="https://github.com/charlesq34/pointnet2">here</a>.

### Todo

- Add a demo script to run inference of Frustum PointNets based on raw input data.
- Add related scripts for SUNRGBD dataset
