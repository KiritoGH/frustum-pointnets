# -*- coding: utf-8 -*-
''' Provider class and helper functions for Frustum PointNets.

Author: Charles R. Qi
Date: September 2017
'''
from __future__ import print_function

# import _pickle as pickle  # 原来是import cPickle，Python3改成import _pickle
import cPickle as pickle
import importlib
import sys
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from box_util import box3d_iou
from model_util import g_type2class, g_class2type, g_type2onehotclass
from model_util import g_type_mean_size
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER

try:
    raw_input  # Python 2
except NameError:
    raw_input = input  # Python 3

# import sys
# importlib.reload(sys)
# sys.setdefaultencoding('utf8')


# 绕Y轴旋转点云
def rotate_pc_along_y(pc, rot_angle):
    '''
    Input:
        pc: numpy array (N,C), first 3 channels are XYZ
            z is facing forward, x is left ward, y is downward
        rot_angle: rad scalar
    Output:
        pc: updated pc with XYZ rotated
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
    return pc


# 将连续角度转换为离散类别和残差
def angle2class(angle, num_class):
    ''' Convert continuous angle to discrete class and residual.

    Input:
        angle: rad scalar, from 0-2pi (or -pi~pi), class center at
            0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        num_class: int scalar, number of classes N
    Output:
        class_id, int, among 0,1,...,N-1
        residual_angle: float, a number such that
            class*(2pi/N) + residual_angle = angle
    '''
    angle = angle % (2 * np.pi)
    assert (angle >= 0 and angle <= 2 * np.pi)
    angle_per_class = 2 * np.pi / float(num_class)  # 角度分辨率
    shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)     # 将类别中心移动至0,1*cta,...(N-1)*cta
    class_id = int(shifted_angle / angle_per_class)                 # 0,1,...N-1
    residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)  # 计算角度离所属类别中心的距离
    return class_id, residual_angle


# 将角度类别还原为角度
def class2angle(pred_cls, residual, num_class, to_label_format=True):
    ''' Inverse function to angle2class.
    If to_label_format, adjust angle to the range as in labels.
    '''
    angle_per_class = 2 * np.pi / float(num_class)
    angle_center = pred_cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle > np.pi:   # -pi~pi
        angle = angle - 2 * np.pi
    return angle


# 将三维边框转换到对应模板类别，并计算残差
def size2class(size, type_name):
    ''' Convert 3D bounding box size to template class and residuals.
    todo (rqi): support multiple size clusters per type.
 
    Input:
        size: numpy array of shape (3,) for (l,w,h)
        type_name: string
    Output:
        size_class: int scalar
        size_residual: numpy array of shape (3,)
    '''
    size_class = g_type2class[type_name]
    size_residual = size - g_type_mean_size[type_name]
    return size_class, size_residual


# 根据模板类别和残差，还原三维边框
def class2size(pred_cls, residual):
    ''' Inverse function to size2class. '''
    mean_size = g_type_mean_size[g_class2type[pred_cls]]
    return mean_size + residual


# 数据集类，从pickle文件载入准备好的KITTI数据
class FrustumDataset(object):
    ''' Dataset class for Frustum PointNets training/evaluation.
    Load prepared KITTI data from pickled files, return individual data element
    [optional] along with its annotations.
    '''
    # 初始化
    def __init__(self, npoints, split,
                 random_flip=False, random_shift=False, rotate_to_center=False,
                 overwritten_data_path=None, from_rgb_detection=False, one_hot=False):
        '''
        Input:
            npoints: int scalar, number of points for frustum point cloud.
            split: string, train or val
            random_flip: bool, in 50% randomly flip the point cloud
                in left and right (after the frustum rotation if any)
            random_shift: bool, if True randomly shift the point cloud
                back and forth by a random distance
            rotate_to_center: bool, whether to do frustum rotation
            overwritten_data_path: string, specify pickled file path.
                if None, use default path (with the split)
            from_rgb_detection: bool, if True we assume we do not have
                groundtruth, just return data elements.
            one_hot: bool, if True, return one hot vector
        '''
        self.npoints = npoints  # int标量，截锥体点云的点数
        self.random_flip = random_flip  # bool，50%的概率对点云进行左右翻转
        self.random_shift = random_shift    # bool，前后随机偏移点云
        self.rotate_to_center = rotate_to_center    # bool，是否做截锥体旋转
        self.one_hot = one_hot  # bool，返回独热编码向量
        if overwritten_data_path is None:   # 没有数据路径时，采用默认路径
            overwritten_data_path = os.path.join(ROOT_DIR, 'kitti/frustum_carpedcyc_%s.pickle' % (split))
            print(overwritten_data_path)

        self.from_rgb_detection = from_rgb_detection    # bool，True表明数据从RGB观测器获取，即没有真值，仅返回数据元素
        print('load starts')
        if from_rgb_detection:
            with open(overwritten_data_path, 'rb') as fp:   # rb代表只读，二进制格式
                self.id_list = pickle.load(fp)
                self.box2d_list = pickle.load(fp)
                self.input_list = pickle.load(fp)
                self.type_list = pickle.load(fp)
                # frustum_angle is clockwise angle from positive x-axis
                self.frustum_angle_list = pickle.load(fp)
                self.prob_list = pickle.load(fp)
        else:
            with open(overwritten_data_path, 'rb') as fp:
                self.id_list = pickle.load(fp)
                self.box2d_list = pickle.load(fp)
                self.box3d_list = pickle.load(fp)
                self.input_list = pickle.load(fp)
                self.label_list = pickle.load(fp)
                self.type_list = pickle.load(fp)
                self.heading_list = pickle.load(fp)
                self.size_list = pickle.load(fp)
                # frustum_angle is clockwise angle from positive x-axis
                self.frustum_angle_list = pickle.load(fp)
        print('load ends')

    def __len__(self):
        return len(self.input_list)

    # 获取索引对应的数据元素
    # 当数据由RGB检测器获取时，返回点云数据、旋转角度、概率列表、独热编码向量
    # 数据从数据集真值获取时，返回点云数据、分割标签、边框中心、朝向角类别、朝向角残差、尺寸类别、尺寸残差、旋转角、独热编码向量
    def __getitem__(self, index):
        ''' Get index-th element from the picked file dataset. '''
        # ------------------------------ INPUTS ----------------------------
        rot_angle = self.get_center_view_rot_angle(index)

        # 计算独热编码向量
        if self.one_hot:
            cls_type = self.type_list[index]
            assert (cls_type in ['Car', 'Pedestrian', 'Cyclist'])
            one_hot_vec = np.zeros((3))
            one_hot_vec[g_type2onehotclass[cls_type]] = 1

        # 获取点云
        if self.rotate_to_center:   # 进行截锥体旋转
            point_set = self.get_center_view_point_set(index)
        else:
            point_set = self.input_list[index]
        # 重采样，从点云数据中随机选取设定点数
        choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        point_set = point_set[choice, :]

        if self.from_rgb_detection:     # 数据由RGB检测器获得，返回点云数据、旋转角度、概率列表
            if self.one_hot:
                return point_set, rot_angle, self.prob_list[index], one_hot_vec
            else:
                return point_set, rot_angle, self.prob_list[index]

        # ------------------------------ LABELS ----------------------------
        # 数据由数据集真值获得
        # 分割标签
        seg = self.label_list[index]
        seg = seg[choice]

        # 3D边框中心
        if self.rotate_to_center:
            box3d_center = self.get_center_view_box3d_center(index)
        else:
            box3d_center = self.get_box3d_center(index)

        # 朝向角
        if self.rotate_to_center:
            heading_angle = self.heading_list[index] - rot_angle
        else:
            heading_angle = self.heading_list[index]

        # 尺寸
        size_class, size_residual = size2class(self.size_list[index],
                                               self.type_list[index])

        # 数据增强，左右翻转、随机偏移
        if self.random_flip:
            # note: rot_angle won't be correct if we have random_flip
            # so do not use it in case of random flipping.
            if np.random.random() > 0.5:  # 50% chance flipping
                point_set[:, 0] *= -1
                box3d_center[0] *= -1
                heading_angle = np.pi - heading_angle
        if self.random_shift:
            dist = np.sqrt(np.sum(box3d_center[0] ** 2 + box3d_center[1] ** 2))
            shift = np.clip(np.random.randn() * dist * 0.05, dist * 0.8, dist * 1.2)
            point_set[:, 2] += shift
            box3d_center[2] += shift

        # 根据设定的朝向角类别数进行分类，计算残差
        angle_class, angle_residual = angle2class(heading_angle, NUM_HEADING_BIN)

        if self.one_hot:
            return point_set, seg, box3d_center, angle_class, angle_residual, \
                   size_class, size_residual, rot_angle, one_hot_vec
        else:
            return point_set, seg, box3d_center, angle_class, angle_residual, \
                   size_class, size_residual, rot_angle

    # 根据索引获取截锥体旋转角度，+pi/2偏移         !!!为什么是这么转换的
    def get_center_view_rot_angle(self, index):
        ''' Get the frustum rotation angle, it is shifted by pi/2 so that it
        can be directly used to adjust GT heading angle '''
        return np.pi / 2.0 + self.frustum_angle_list[index]

    # 根据索引找到边框的对角点，计算边框中心
    def get_box3d_center(self, index):
        ''' Get the center (XYZ) of 3D bounding box. '''
        box3d_center = (self.box3d_list[index][0, :] + self.box3d_list[index][6, :]) / 2.0
        return box3d_center

    # 根据索引获取截锥体旋转角度，然后对3D边框中心进行旋转
    def get_center_view_box3d_center(self, index):
        ''' Frustum rotation of 3D bounding box center. '''
        box3d_center = (self.box3d_list[index][0, :] + self.box3d_list[index][6, :]) / 2.0
        # 先对中心进行维度拓展变成向量(1,C)，旋转之后再返回(C,)
        return rotate_pc_along_y(np.expand_dims(box3d_center, 0), self.get_center_view_rot_angle(index)).squeeze()

    # 对3D边框(8,C)进行旋转
    def get_center_view_box3d(self, index):
        ''' Frustum rotation of 3D bounding box corners. '''
        box3d = self.box3d_list[index]
        box3d_center_view = np.copy(box3d)
        return rotate_pc_along_y(box3d_center_view, self.get_center_view_rot_angle(index))

    # 旋转点云
    def get_center_view_point_set(self, index):
        ''' Frustum rotation of point clouds.
        NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
        '''
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(self.input_list[index])
        return rotate_pc_along_y(point_set, self.get_center_view_rot_angle(index))


# ----------------------------------
# Helper functions for evaluation
# ----------------------------------
# 计算3D边框的8个顶点
def get_3d_box(box_size, heading_angle, center):
    ''' Calculate 3D bounding box corners from its parameterization.

    Input:
        box_size: tuple of (l,w,h)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''

    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])

    R = roty(heading_angle)
    l, w, h = box_size
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = np.transpose(corners_3d)
    return corners_3d


# 根据网络输出和标签真值，计算3D边框的交并比IoU
def compute_box3d_iou(center_pred,
                      heading_logits, heading_residuals,
                      size_logits, size_residuals,
                      center_label,
                      heading_class_label, heading_residual_label,
                      size_class_label, size_residual_label):
    ''' Compute 3D bounding box IoU from network output and labels.
    All inputs are numpy arrays.

    Inputs:
        center_pred: (B,3)
        heading_logits: (B,NUM_HEADING_BIN)
        heading_residuals: (B,NUM_HEADING_BIN)
        size_logits: (B,NUM_SIZE_CLUSTER)
        size_residuals: (B,NUM_SIZE_CLUSTER,3)
        center_label: (B,3)
        heading_class_label: (B,)
        heading_residual_label: (B,)
        size_class_label: (B,)
        size_residual_label: (B,3)
    Output:
        iou2ds: (B,) birdeye view oriented 2d box ious
        iou3ds: (B,) 3d box ious
    '''
    batch_size = heading_logits.shape[0]
    heading_class = np.argmax(heading_logits, 1)  # 获得朝向角类别(B,)
    # 获得朝向角类别对应的残差(B,)
    heading_residual = np.array([heading_residuals[i, heading_class[i]] for i in range(batch_size)])
    size_class = np.argmax(size_logits, 1)  # 尺寸类别(B,)
    # 获得尺寸类别对应的残差(B,)
    size_residual = np.vstack([size_residuals[i, size_class[i], :] for i in range(batch_size)])

    iou2d_list = []
    iou3d_list = []
    for i in range(batch_size):
        heading_angle = class2angle(heading_class[i], heading_residual[i], NUM_HEADING_BIN)     # 还原朝向角（预测值）
        box_size = class2size(size_class[i], size_residual[i])  # 还原边框尺寸（预测值）
        corners_3d = get_3d_box(box_size, heading_angle, center_pred[i])    # 计算边框的8个顶点（预测值）
        # 还原朝向角真值
        heading_angle_label = class2angle(heading_class_label[i], heading_residual_label[i], NUM_HEADING_BIN)
        box_size_label = class2size(size_class_label[i], size_residual_label[i])    # 还原边框尺寸真值
        corners_3d_label = get_3d_box(box_size_label, heading_angle_label, center_label[i])     # 计算边框顶点真值

        iou_3d, iou_2d = box3d_iou(corners_3d, corners_3d_label)    # 根据顶点预测值和真值，计算二维、三维IoU
        iou3d_list.append(iou_3d)
        iou2d_list.append(iou_2d)
    return np.array(iou2d_list, dtype=np.float32), np.array(iou3d_list, dtype=np.float32)


# 将预测的边框参数转换到标签形式
def from_prediction_to_label_format(center, angle_class, angle_res, size_class, size_res, rot_angle):
    ''' Convert predicted box parameters to label format. '''
    l, w, h = class2size(size_class, size_res)
    ry = class2angle(angle_class, angle_res, NUM_HEADING_BIN) + rot_angle
    tx, ty, tz = rotate_pc_along_y(np.expand_dims(center, 0), -rot_angle).squeeze()
    ty += h / 2.0
    return h, w, l, tx, ty, tz, ry


if __name__ == '__main__':
    import mayavi.mlab as mlab

    sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
    from viz_util import draw_lidar, draw_gt_boxes3d

    median_list = []
    dataset = FrustumDataset(1024, split='val',
                             rotate_to_center=True, random_flip=True, random_shift=True)
    for i in range(len(dataset)):
        data = dataset[i]
        print(('Center: ', data[2], \
               'angle_class: ', data[3], 'angle_res:', data[4], \
               'size_class: ', data[5], 'size_residual:', data[6], \
               'real_size:', g_type_mean_size[g_class2type[data[5]]] + data[6]))
        print(('Frustum angle: ', dataset.frustum_angle_list[i]))
        median_list.append(np.median(data[0][:, 0]))
        print((data[2], dataset.box3d_list[i], median_list[-1]))
        box3d_from_label = get_3d_box(class2size(data[5], data[6]), class2angle(data[3], data[4], 12), data[2])

        ps = data[0]
        seg = data[1]
        fig = mlab.figure(figure=None, bgcolor=(0.4, 0.4, 0.4), fgcolor=None, engine=None, size=(1000, 500))
        mlab.points3d(ps[:, 0], ps[:, 1], ps[:, 2], seg, mode='point', colormap='gnuplot', scale_factor=1, figure=fig)
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2, figure=fig)
        draw_gt_boxes3d([box3d_from_label], fig, color=(1, 0, 0))
        mlab.orientation_axes()
        raw_input()
    print(np.mean(np.abs(median_list)))
