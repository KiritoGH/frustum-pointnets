# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import tf_util

# -----------------
# Global Constants
# -----------------

NUM_HEADING_BIN = 12    # 设定12种朝向，每30°一种
NUM_SIZE_CLUSTER = 8    # 设定8种尺寸类型
NUM_OBJECT_POINT = 512  # 物体点数
# 字典，str:int
g_type2class = {'Car': 0, 'Van': 1, 'Truck': 2, 'Pedestrian': 3,
                'Person_sitting': 4, 'Cyclist': 5, 'Tram': 6, 'Misc': 7}
# 字典，int:str
g_class2type = {g_type2class[t]: t for t in g_type2class}
# 字典，用于转换为独热编码
g_type2onehotclass = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
# 8种类型的平均尺寸
g_type_mean_size = {'Car': np.array([3.88311640418, 1.62856739989, 1.52563191462]),
                    'Van': np.array([5.06763659, 1.9007158, 2.20532825]),
                    'Truck': np.array([10.13586957, 2.58549199, 3.2520595]),
                    'Pedestrian': np.array([0.84422524, 0.66068622, 1.76255119]),
                    'Person_sitting': np.array([0.80057803, 0.5983815, 1.27450867]),
                    'Cyclist': np.array([1.76282397, 0.59706367, 1.73698127]),
                    'Tram': np.array([16.17150617, 2.53246914, 3.53079012]),
                    'Misc': np.array([3.64300781, 1.54298177, 1.92320313])}
g_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3))  # 初始化，(NS,3)
# 将8种类型的尺寸赋值给g_mean_size_arr
for i in range(NUM_SIZE_CLUSTER):
    g_mean_size_arr[i, :] = g_type_mean_size[g_class2type[i]]


# -----------------
# TF Functions Helpers
# -----------------

# 根据mask获取物体点云
# 输入点云、mask、最大点数
# 输出物体点云和点云索引
def tf_gather_object_pc(point_cloud, mask, npoints=512):
    ''' Gather object point clouds according to predicted masks.
    Input:
        point_cloud: TF tensor in shape (B,N,C)
        mask: TF tensor in shape (B,N) of 0 (not pick) or 1 (pick)
        npoints: int scalar, maximum number of points to keep (default: 512)
    Output:
        object_pc: TF tensor in shape (B,npoint,C)
        indices: TF int tensor in shape (B,npoint,2)
    '''

    def mask_to_indices(mask):
        indices = np.zeros((mask.shape[0], npoints, 2), dtype=np.int32)
        for i in range(mask.shape[0]):
            # np.where(condition,x,y)，满足condition，输出x，否则输出y
            # np.where(condition)，输出满足condition(即非0)元素的坐标
            # 返回元组，所以最后加了个[0]变为numpy
            pos_indices = np.where(mask[i, :] > 0.5)[0]
            # print(pos_indices.shape)
            # pos_indices为空即第i个样本没有mask大于0.5，则跳过
            if len(pos_indices) > 0:
                if len(pos_indices) > npoints:  # 点数多于最大点数，从中随机选取最大点数个点（不放回）
                    choice = np.random.choice(len(pos_indices), npoints, replace=False)
                else:                           # 否则，保留点的同时，再随机挑选一些补足空余（放回）
                    choice = np.random.choice(len(pos_indices), npoints - len(pos_indices), replace=True)
                    choice = np.concatenate((np.arange(len(pos_indices)), choice))
                np.random.shuffle(choice)       # 打乱索引的索引
                indices[i, :, 1] = pos_indices[choice]
            indices[i, :, 0] = i
        return indices

    # tf.py_func接收tensor，转换为numpy进行自定义函数操作，最后将输出转化为tensor返回。
    # tensor没有实际值，不能进行判断操作
    indices = tf.py_func(mask_to_indices, [mask], tf.int32)
    object_pc = tf.gather_nd(point_cloud, indices)  # 根据索引从点云中收集对应数据
    return object_pc, indices


# 计算3D边框的8个顶点坐标的基础函数
# 输入中心(N,3)、朝向角(N,)、尺寸(N,3)
# 输出顶点坐标(N,8,3)
def get_box3d_corners_helper(centers, headings, sizes):
    """ TF layer. Input: (N,3), (N,), (N,3), Output: (N,8,3) """
    # print '-----', centers
    N = centers.get_shape()[0].value
    # sizes沿第1（二）维度划分为三部分，即l,w,h
    l = tf.slice(sizes, [0, 0], [-1, 1])  # (N,1)
    w = tf.slice(sizes, [0, 1], [-1, 1])  # (N,1)
    h = tf.slice(sizes, [0, 2], [-1, 1])  # (N,1)
    # 8个顶点相对于中心的位置坐标
    x_corners = tf.concat([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], axis=1)  # (N,8)
    y_corners = tf.concat([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2], axis=1)  # (N,8)
    z_corners = tf.concat([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], axis=1)  # (N,8)
    corners = tf.concat([tf.expand_dims(x_corners, 1), tf.expand_dims(y_corners, 1), tf.expand_dims(z_corners, 1)],
                        axis=1)  # (N,3,8)

    c = tf.cos(headings)
    s = tf.sin(headings)
    ones = tf.ones([N], dtype=tf.float32)
    zeros = tf.zeros([N], dtype=tf.float32)
    row1 = tf.stack([c, zeros, s], axis=1)  # (N,3)
    row2 = tf.stack([zeros, ones, zeros], axis=1)
    row3 = tf.stack([-s, zeros, c], axis=1)
    # y轴旋转矩阵
    R = tf.concat([tf.expand_dims(row1, 1), tf.expand_dims(row2, 1), tf.expand_dims(row3, 1)], axis=1)  # (N,3,3)
    # 对8个顶点根据朝向进行旋转，计算绝对位置坐标
    corners_3d = tf.matmul(R, corners)  # (N,3,8)
    corners_3d += tf.tile(tf.expand_dims(centers, 2), [1, 1, 8])  # (N,3,8)
    corners_3d = tf.transpose(corners_3d, perm=[0, 2, 1])  # (N,8,3)
    return corners_3d


# 利用基础函数，计算B个中心NH×NS种情况即N=B×NH×NS
# 输入中心(B,3)、朝向角残差(B,NH)、尺寸残差(B,NS,3)
# 输出所有顶点坐标向量(B,NH,NS,8,3)
def get_box3d_corners(center, heading_residuals, size_residuals):
    """ TF layer.
    Inputs:
        center: (B,3)
        heading_residuals: (B,NH)
        size_residuals: (B,NS,3)
    Outputs:
        box3d_corners: (B,NH,NS,8,3) tensor
    """
    batch_size = center.get_shape()[0].value
    # 2pi划分为NH份，加上朝向角残差
    heading_bin_centers = tf.constant(np.arange(0, 2 * np.pi, 2 * np.pi / NUM_HEADING_BIN), dtype=tf.float32)
    headings = heading_residuals + tf.expand_dims(heading_bin_centers, 0)  # (B,NH)
    # NS种类型尺寸加上残差
    mean_sizes = tf.expand_dims(tf.constant(g_mean_size_arr, dtype=tf.float32), 0)  # (1,NS,3)
    sizes = mean_sizes + size_residuals  # (B,NS,3)
    sizes = tf.tile(tf.expand_dims(sizes, 1), [1, NUM_HEADING_BIN, 1, 1])  # (B,NH,NS,3)
    headings = tf.tile(tf.expand_dims(headings, -1), [1, 1, NUM_SIZE_CLUSTER])  # (B,NH,NS)
    centers = tf.tile(tf.expand_dims(tf.expand_dims(center, 1), 1),
                      [1, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 1])  # (B,NH,NS,3)
    # 所有情况N
    N = batch_size * NUM_HEADING_BIN * NUM_SIZE_CLUSTER
    corners_3d = get_box3d_corners_helper(tf.reshape(centers, [N, 3]), tf.reshape(headings, [N]),
                                          tf.reshape(sizes, [N, 3]))

    return tf.reshape(corners_3d, [batch_size, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 8, 3])


def huber_loss(error, delta):
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic ** 2 + delta * linear
    return tf.reduce_mean(losses)


# 将输出张量拆分成几个不同部分，添加到end_points
def parse_output_to_tensors(output, end_points):
    ''' Parse batch output to separate tensors (added to end_points)
    Input:
        output: TF tensor in shape (B,3+2*NUM_HEADING_BIN+4*NUM_SIZE_CLUSTER)
        end_points: dict
    Output:
        end_points: dict (updated)
    '''
    batch_size = output.get_shape()[0].value
    center = tf.slice(output, [0, 0], [-1, 3])
    end_points['center_boxnet'] = center

    heading_scores = tf.slice(output, [0, 3], [-1, NUM_HEADING_BIN])    # 朝向角得分
    heading_residuals_normalized = tf.slice(output, [0, 3 + NUM_HEADING_BIN],
                                            [-1, NUM_HEADING_BIN])
    end_points['heading_scores'] = heading_scores  # BxNUM_HEADING_BIN
    end_points['heading_residuals_normalized'] = \
        heading_residuals_normalized  # BxNUM_HEADING_BIN 归一化到(-1,1)
    end_points['heading_residuals'] = \
        heading_residuals_normalized * (np.pi / NUM_HEADING_BIN)  # BxNUM_HEADING_BIN 重新转换到(-pi/NH,pi/NH)

    size_scores = tf.slice(output, [0, 3 + NUM_HEADING_BIN * 2],
                           [-1, NUM_SIZE_CLUSTER])  # BxNUM_SIZE_CLUSTER    尺寸得分
    size_residuals_normalized = tf.slice(output,
                                         [0, 3 + NUM_HEADING_BIN * 2 + NUM_SIZE_CLUSTER], [-1, NUM_SIZE_CLUSTER * 3])
    size_residuals_normalized = tf.reshape(size_residuals_normalized,
                                           [batch_size, NUM_SIZE_CLUSTER, 3])  # BxNUM_SIZE_CLUSTERx3
    end_points['size_scores'] = size_scores
    end_points['size_residuals_normalized'] = size_residuals_normalized
    end_points['size_residuals'] = size_residuals_normalized * \
                                   tf.expand_dims(tf.constant(g_mean_size_arr, dtype=tf.float32), 0)    # 尺寸还原

    return end_points


# --------------------------------------
# Shared subgraphs for v1 and v2 models
# --------------------------------------

# 创建占位符
def placeholder_inputs(batch_size, num_point):
    ''' Get useful placeholder tensors.
    Input:
        batch_size: scalar int
        num_point: scalar int
    Output:
        TF placeholders for inputs and ground truths
    '''
    pointclouds_pl = tf.placeholder(tf.float32,
                                    shape=(batch_size, num_point, 4))
    one_hot_vec_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))

    # labels_pl is for segmentation label
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    centers_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))
    heading_class_label_pl = tf.placeholder(tf.int32, shape=(batch_size,))
    heading_residual_label_pl = tf.placeholder(tf.float32, shape=(batch_size,))
    size_class_label_pl = tf.placeholder(tf.int32, shape=(batch_size,))
    size_residual_label_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))

    return pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, \
           heading_class_label_pl, heading_residual_label_pl, \
           size_class_label_pl, size_residual_label_pl


# 用预测的三维掩膜选取点云，并将坐标系转换到点云中心
# 输入点云(B,N,C)，C表征点信息的通道，如三维坐标，强度等
# logits为经过实例分割网络后的输出(B,N,2)
# xyz_only为是否只返回XYZ通道
# 输出物体点云(B,M,3)，为了简化，只保留了XYZ，M是物体点数，作为一个超参数
# 输出mask后点云的坐标中心(B,3)，end_points['mask'] = mask
def point_cloud_masking(point_cloud, logits, end_points, xyz_only=True):
    ''' Select point cloud with predicted 3D mask,
    translate coordinates to the masked points centroid.
    
    Input:
        point_cloud: TF tensor in shape (B,N,C)
        logits: TF tensor in shape (B,N,2)
        end_points: dict
        xyz_only: boolean, if True only return XYZ channels
    Output:
        object_point_cloud: TF tensor in shape (B,M,3)
            for simplicity we only keep XYZ here
            M = NUM_OBJECT_POINT as a hyper-parameter
        mask_xyz_mean: TF tensor in shape (B,3)
    '''
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    # 切片tf.slice(input,begin,size,name),begin指定开始位置,size指定切片大小,-1为从开始一直到结束
    # 比较(B,N,0)(B,N,1)，并转换为float
    mask = tf.slice(logits, [0, 0, 0], [-1, -1, 1]) < tf.slice(logits, [0, 0, 1], [-1, -1, 1])
    mask = tf.to_float(mask)  # BxNx1
    # 求和tf.reduce_sum(input,axis,keep_dims),axis指定求和的维度,keep_dims决定是否保留原始维度
    # 先对mask的第1维度求和，保留维度，再进行复制平铺，得到(B,1,3)即掩膜选取下的点数
    mask_count = tf.tile(tf.reduce_sum(mask, axis=1, keep_dims=True), [1, 1, 3])  # Bx1x3
    # 对输入点云进行切片，选取三维坐标
    point_cloud_xyz = tf.slice(point_cloud, [0, 0, 0], [-1, -1, 3])  # BxNx3
    # 复制平铺掩膜为(B,N,3)，作用在点云坐标上，对N个点的三维坐标进行求和，保留维度，求平均值，得到掩膜后点云的坐标中心
    mask_xyz_mean = tf.reduce_sum(tf.tile(mask, [1, 1, 3]) * point_cloud_xyz,
                                  axis=1, keep_dims=True) / tf.maximum(mask_count, 1)  # Bx1x3
    mask = tf.squeeze(mask, axis=[2])  # BxN
    end_points['mask'] = mask

    # 将输入坐标转换到掩膜点云坐标中心
    point_cloud_xyz_stage1 = point_cloud_xyz - tf.tile(mask_xyz_mean, [1, num_point, 1])
    # 点云只返回XYZ坐标（已转换到掩膜点云坐标中心），或者拼接上其它通道
    if xyz_only:
        point_cloud_stage1 = point_cloud_xyz_stage1
    else:
        point_cloud_features = tf.slice(point_cloud, [0, 0, 3], [-1, -1, -1])
        point_cloud_stage1 = tf.concat([point_cloud_xyz_stage1, point_cloud_features], axis=-1)
    # 获得点云数据的通道数
    num_channels = point_cloud_stage1.get_shape()[2].value
    # 根据mask和物体点数获取物体点云
    object_point_cloud, _ = tf_gather_object_pc(point_cloud_stage1, mask, NUM_OBJECT_POINT)
    object_point_cloud.set_shape([batch_size, NUM_OBJECT_POINT, num_channels])

    return object_point_cloud, tf.squeeze(mask_xyz_mean, axis=1), end_points


# 中心残差回归网络
# 输入3D mask坐标系下的物体点云(B,M,C)，独热向量
# 输出预测中心残差(B,3)
def get_center_regression_net(object_point_cloud, one_hot_vec,
                              is_training, bn_decay, end_points):
    ''' Regression network for center delta. a.k.a. T-Net.
    Input:
        object_point_cloud: TF tensor in shape (B,M,C)
            point clouds in 3D mask coordinate
        one_hot_vec: TF tensor in shape (B,3)
            length-3 vectors indicating predicted object type
    Output:
        predicted_center: TF tensor in shape (B,3)
    '''
    num_point = object_point_cloud.get_shape()[1].value
    net = tf.expand_dims(object_point_cloud, 2)                                 # B×M×1×C
    # MLP(128,128,256)，此处与论文不一致！！！
    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv-reg1-stage1', bn_decay=bn_decay)           # B×M×1×128
    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv-reg2-stage1', bn_decay=bn_decay)           # B×M×1×128
    net = tf_util.conv2d(net, 256, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv-reg3-stage1', bn_decay=bn_decay)           # B×M×1×256
    # max pool
    net = tf_util.max_pool2d(net, [num_point, 1],
                             padding='VALID', scope='maxpool-stage1')           # B×1×1×256
    net = tf.squeeze(net, axis=[1, 2])                                          # B×256
    net = tf.concat([net, one_hot_vec], axis=1)                                 # B×259
    # FCs(256,128,3)
    net = tf_util.fully_connected(net, 256, scope='fc1-stage1', bn=True,
                                  is_training=is_training, bn_decay=bn_decay)   # B×256
    net = tf_util.fully_connected(net, 128, scope='fc2-stage1', bn=True,
                                  is_training=is_training, bn_decay=bn_decay)   # B×128
    predicted_center = tf_util.fully_connected(net, 3, activation_fn=None,
                                               scope='fc3-stage1')              # B×3
    return predicted_center, end_points


# 计算损失函数
def get_loss(mask_label, center_label,
             heading_class_label, heading_residual_label,
             size_class_label, size_residual_label,
             end_points,
             corner_loss_weight=10.0,
             box_loss_weight=1.0):
    ''' Loss functions for 3D object detection.
    Input:
        mask_label: TF int32 tensor in shape (B,N)
        center_label: TF tensor in shape (B,3)
        heading_class_label: TF int32 tensor in shape (B,) 
        heading_residual_label: TF tensor in shape (B,) 
        size_class_label: TF tensor int32 in shape (B,)
        size_residual_label: TF tensor tensor in shape (B,)
        end_points: dict, outputs from our model
        corner_loss_weight: float scalar
        box_loss_weight: float scalar
    Output:
        total_loss: TF scalar tensor
            the total_loss is also added to the losses collection
    '''
    # 分割网络输出logits计算3D分割损失
    mask_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=end_points['mask_logits'], labels=mask_label))
    tf.summary.scalar('3d mask loss', mask_loss)

    # 中心回归损失
    center_dist = tf.norm(center_label - end_points['center'], axis=-1)
    center_loss = huber_loss(center_dist, delta=2.0)
    tf.summary.scalar('center loss', center_loss)
    stage1_center_dist = tf.norm(center_label -
                                 end_points['stage1_center'], axis=-1)
    stage1_center_loss = huber_loss(stage1_center_dist, delta=1.0)
    tf.summary.scalar('stage1 center loss', stage1_center_loss)

    # 朝向损失
    heading_class_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=end_points['heading_scores'], labels=heading_class_label))
    tf.summary.scalar('heading class loss', heading_class_loss)

    hcls_onehot = tf.one_hot(heading_class_label,
                             depth=NUM_HEADING_BIN,
                             on_value=1, off_value=0, axis=-1)  # BxNUM_HEADING_BIN
    heading_residual_normalized_label = \
        heading_residual_label / (np.pi / NUM_HEADING_BIN)
    heading_residual_normalized_loss = huber_loss(tf.reduce_sum(
        end_points['heading_residuals_normalized'] * tf.to_float(hcls_onehot), axis=1) -
                                                  heading_residual_normalized_label, delta=1.0)
    tf.summary.scalar('heading residual normalized loss',
                      heading_residual_normalized_loss)

    # 尺寸损失
    size_class_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=end_points['size_scores'], labels=size_class_label))
    tf.summary.scalar('size class loss', size_class_loss)

    scls_onehot = tf.one_hot(size_class_label,
                             depth=NUM_SIZE_CLUSTER,
                             on_value=1, off_value=0, axis=-1)  # BxNUM_SIZE_CLUSTER
    scls_onehot_tiled = tf.tile(tf.expand_dims(
        tf.to_float(scls_onehot), -1), [1, 1, 3])  # BxNUM_SIZE_CLUSTERx3
    predicted_size_residual_normalized = tf.reduce_sum(
        end_points['size_residuals_normalized'] * scls_onehot_tiled, axis=[1])  # Bx3

    mean_size_arr_expand = tf.expand_dims(
        tf.constant(g_mean_size_arr, dtype=tf.float32), 0)  # 1xNUM_SIZE_CLUSTERx3
    mean_size_label = tf.reduce_sum(
        scls_onehot_tiled * mean_size_arr_expand, axis=[1])  # Bx3
    size_residual_label_normalized = size_residual_label / mean_size_label
    size_normalized_dist = tf.norm(
        size_residual_label_normalized - predicted_size_residual_normalized,
        axis=-1)
    size_residual_normalized_loss = huber_loss(size_normalized_dist, delta=1.0)
    tf.summary.scalar('size residual normalized loss',
                      size_residual_normalized_loss)

    # 顶点损失
    # We select the predicted corners corresponding to the 
    # GT heading bin and size cluster.
    corners_3d = get_box3d_corners(end_points['center'],
                                   end_points['heading_residuals'],
                                   end_points['size_residuals'])  # (B,NH,NS,8,3)
    gt_mask = tf.tile(tf.expand_dims(hcls_onehot, 2), [1, 1, NUM_SIZE_CLUSTER]) * \
              tf.tile(tf.expand_dims(scls_onehot, 1), [1, NUM_HEADING_BIN, 1])  # (B,NH,NS)
    corners_3d_pred = tf.reduce_sum(
        tf.to_float(tf.expand_dims(tf.expand_dims(gt_mask, -1), -1)) * corners_3d,
        axis=[1, 2])  # (B,8,3)

    heading_bin_centers = tf.constant(
        np.arange(0, 2 * np.pi, 2 * np.pi / NUM_HEADING_BIN), dtype=tf.float32)  # (NH,)
    heading_label = tf.expand_dims(heading_residual_label, 1) + \
                    tf.expand_dims(heading_bin_centers, 0)  # (B,NH)
    heading_label = tf.reduce_sum(tf.to_float(hcls_onehot) * heading_label, 1)
    mean_sizes = tf.expand_dims(
        tf.constant(g_mean_size_arr, dtype=tf.float32), 0)  # (1,NS,3)
    size_label = mean_sizes + \
                 tf.expand_dims(size_residual_label, 1)  # (1,NS,3) + (B,1,3) = (B,NS,3)
    size_label = tf.reduce_sum(
        tf.expand_dims(tf.to_float(scls_onehot), -1) * size_label, axis=[1])  # (B,3)
    corners_3d_gt = get_box3d_corners_helper(
        center_label, heading_label, size_label)  # (B,8,3)
    corners_3d_gt_flip = get_box3d_corners_helper(
        center_label, heading_label + np.pi, size_label)  # (B,8,3)

    corners_dist = tf.minimum(tf.norm(corners_3d_pred - corners_3d_gt, axis=-1),
                              tf.norm(corners_3d_pred - corners_3d_gt_flip, axis=-1))
    corners_loss = huber_loss(corners_dist, delta=1.0)
    tf.summary.scalar('corners loss', corners_loss)

    # 权重求和
    total_loss = mask_loss + box_loss_weight * (center_loss +
                                                heading_class_loss + size_class_loss +
                                                heading_residual_normalized_loss * 20 +
                                                size_residual_normalized_loss * 20 +
                                                stage1_center_loss +
                                                corner_loss_weight * corners_loss)
    tf.add_to_collection('losses', total_loss)

    return total_loss


# 自己测试
if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 1024, 4))
        mask = tf.ones((32, 1024))
        object_pc, indices = tf_gather_object_pc(inputs, mask)
        print('end')

