import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

import matplotlib.pyplot as plt

tf.enable_eager_execution()

from waymo_open_dataset.protos import segmentation_metrics_pb2
from waymo_open_dataset.protos import segmentation_submission_pb2

# Data location. Please edit.

# A tfrecord containing tf.Example protos as downloaded from the Waymo dataset
# webpage.

# Replace this path with your own tfrecords.
FILENAME = 'dataset.tfrecord'

"""# Read 3D semantic segmentation labels from Frame proto
 Note that only a subset of the frames have 3d semseg labels.
"""

import sys

dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
cnt = 0
cnt2 = 0
frames = []
all_frames = []
for data in dataset:
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    all_frames.append(frame)
    cnt += 1
    if frame.lasers[0].ri_return1.segmentation_label_compressed:
        frames.append(frame)
        cnt2 += 1
print("cnt:", cnt)
print("cnt2:", cnt2)

frame = frames[0]
print("Context:", frame.context)

print("TimeStamp Micros:", frame.timestamp_micros)

print("Pose:", frame.pose)

"""## Exploring Frame Images Class"""

# plt.figure(figsize = (25, 15))
# for i in range(len(frame.images)):
#     print(sys.getsizeof(frame.images[i].image))
#     img = frame.images[i].image
#     plt.subplot(2, 3, i+1)
#     plt.imshow(tf.image.decode_jpeg(img))

print("Image Pose:", frame.images[0].pose)

print("Image Velocity:", frame.images[0].velocity)

print("Image Shutter:", frame.images[0].shutter)

label_list = []
image_list = []
for frame_ in all_frames:
  for i in range(len(frame_.images)):
    panoptic_image = frame_.images[i].camera_segmentation_label.panoptic_label
    try:
      label = tf.image.decode_png(panoptic_image)
      img = tf.image.decode_jpeg(frame_.images[i].image)
      label_list.append(label)
      image_list.append(img)
    except:
      pass

print(len(label_list))
len(image_list)

"""## Visualizing the Panoptic Labels"""

# import random

# indices = random.sample(range(100), 5)

# plt.figure(figsize = (16, 30))
# for i, index in enumerate(indices):
#   plt.subplot(5, 2, i * 2 + 1)
#   plt.title("Image")
#   plt.imshow(image_list[index])

#   plt.subplot(5, 2, i * 2 + 2)
#   plt.title("Label")
#   plt.imshow(label_list[index][:,:,0])

print("Unique Label Classes:", np.unique(label_list[0].numpy()))

(range_images, camera_projections, segmentation_labels,
 range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(
    frame)

segmentation_labels.keys()

print(segmentation_labels[open_dataset.LaserName.TOP][0].shape.dims)

"""## Lidar data"""

plt.figure(figsize=(64, 20))
def plot_range_image_helper(data, name, layout, vmin = 0, vmax=1, cmap='gray'):
  """Plots range image.

  Args:
    data: range image data
    name: the image title
    layout: plt layout
    vmin: minimum value of the passed data
    vmax: maximum value of the passed data
    cmap: color map
  """
  plt.subplot(*layout)
  plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
  plt.title(name)
  plt.grid(False)
  plt.axis('off')

def get_range_image(laser_name, return_index):
  """Returns range image given a laser name and its return index."""
  return range_images[laser_name][return_index]

def show_range_image(range_image, layout_index_start = 1):
  """Shows range image.

  Args:
    range_image: the range image data from a given lidar of type MatrixFloat.
    layout_index_start: layout offset
  """
  range_image_tensor = tf.convert_to_tensor(range_image.data)
  range_image_tensor = tf.reshape(range_image_tensor, range_image.shape.dims)
  lidar_image_mask = tf.greater_equal(range_image_tensor, 0)
  range_image_tensor = tf.where(lidar_image_mask, range_image_tensor,
                                tf.ones_like(range_image_tensor) * 1e10)
  range_image_range = range_image_tensor[...,0] 
  range_image_intensity = range_image_tensor[...,1]
  range_image_elongation = range_image_tensor[...,2]
  plot_range_image_helper(range_image_range.numpy(), 'range',
                   [8, 1, layout_index_start], vmax=75, cmap='gray')
  plot_range_image_helper(range_image_intensity.numpy(), 'intensity',
                   [8, 1, layout_index_start + 1], vmax=1.5, cmap='gray')
  plot_range_image_helper(range_image_elongation.numpy(), 'elongation',
                   [8, 1, layout_index_start + 2], vmax=1.5, cmap='gray')
frame.lasers.sort(key=lambda laser: laser.name)
# 1st return for TOP sensor
show_range_image(get_range_image(open_dataset.LaserName.TOP, 0), 1)
# 2nd return for TOP sensor

show_range_image(get_range_image(open_dataset.LaserName.TOP, 1), 4)

"""##Point Cloud Conversion and Visualization"""

points, cp_points = frame_utils.convert_range_image_to_point_cloud(
    frame,
    range_images,
    camera_projections,
    range_image_top_pose)
points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
    frame,
    range_images,
    camera_projections,
    range_image_top_pose,
    ri_index=1)

# 3d points in vehicle frame.
points_all = np.concatenate(points, axis=0)
points_all_ri2 = np.concatenate(points_ri2, axis=0)
# camera projection corresponding to each point.
cp_points_all = np.concatenate(cp_points, axis=0)
cp_points_all_ri2 = np.concatenate(cp_points_ri2, axis=0)

"""###Examine number of points in each lidar sensor.

First return.
"""

print(points_all.shape)
print(cp_points_all.shape)
print(points_all[0:2])
for i in range(5):
  print(points[i].shape)
  print(cp_points[i].shape)

"""Second return."""

print(points_all_ri2.shape)
print(cp_points_all_ri2.shape)
print(points_all_ri2[0:2])
for i in range(5):
  print(points_ri2[i].shape)
  print(cp_points_ri2[i].shape)

"""# Visualize Segmentation Labels in Range Images"""

plt.figure(figsize=(64, 20))
def plot_range_image_helper(data, name, layout, vmin = 0, vmax=1, cmap='gray'):
  """Plots range image.

  Args:
    data: range image data
    name: the image title
    layout: plt layout
    vmin: minimum value of the passed data
    vmax: maximum value of the passed data
    cmap: color map
  """
  plt.subplot(*layout)
  plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
  plt.title(name)
  plt.grid(False)
  plt.axis('off')

def get_semseg_label_image(laser_name, return_index):
  """Returns semseg label image given a laser name and its return index."""
  return segmentation_labels[laser_name][return_index]

def show_semseg_label_image(semseg_label_image, layout_index_start = 1):
  """Shows range image.

  Args:
    show_semseg_label_image: the semseg label data of type MatrixInt32.
    layout_index_start: layout offset
  """
  semseg_label_image_tensor = tf.convert_to_tensor(semseg_label_image.data)
  semseg_label_image_tensor = tf.reshape(
      semseg_label_image_tensor, semseg_label_image.shape.dims)
  instance_id_image = semseg_label_image_tensor[...,0] 
  semantic_class_image = semseg_label_image_tensor[...,1]
  plot_range_image_helper(instance_id_image.numpy(), 'instance id',
                   [8, 1, layout_index_start], vmin=-1, vmax=200, cmap='Paired')
  plot_range_image_helper(semantic_class_image.numpy(), 'semantic class',
                   [8, 1, layout_index_start + 1], vmin=0, vmax=22, cmap='tab20')

frame.lasers.sort(key=lambda laser: laser.name)
show_semseg_label_image(get_semseg_label_image(open_dataset.LaserName.TOP, 0), 1)
show_semseg_label_image(get_semseg_label_image(open_dataset.LaserName.TOP, 1), 4)

"""# Point Cloud Conversion and Visualization"""

def convert_range_image_to_point_cloud_labels(frame,
                                              range_images,
                                              segmentation_labels,
                                              ri_index=0):
  """Convert segmentation labels from range images to point clouds.

  Args:
    frame: open dataset frame
    range_images: A dict of {laser_name, [range_image_first_return,
       range_image_second_return]}.
    segmentation_labels: A dict of {laser_name, [range_image_first_return,
       range_image_second_return]}.
    ri_index: 0 for the first return, 1 for the second return.

  Returns:
    point_labels: {[N, 2]} list of 3d lidar points's segmentation labels. 0 for
      points that are not labeled.
  """
  calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
  point_labels = []
  for c in calibrations:
    range_image = range_images[c.name][ri_index]
    range_image_tensor = tf.reshape(
        tf.convert_to_tensor(range_image.data), range_image.shape.dims)
    range_image_mask = range_image_tensor[..., 0] > 0

    if c.name in segmentation_labels:
      sl = segmentation_labels[c.name][ri_index]
      sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data), sl.shape.dims)
      sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
    else:
      num_valid_point = tf.math.reduce_sum(tf.cast(range_image_mask, tf.int32))
      sl_points_tensor = tf.zeros([num_valid_point, 2], dtype=tf.int32)
      
    point_labels.append(sl_points_tensor.numpy())
  return point_labels

points, cp_points = frame_utils.convert_range_image_to_point_cloud(
    frame, range_images, camera_projections, range_image_top_pose)
points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
    frame, range_images, camera_projections, range_image_top_pose, ri_index=1)

point_labels = convert_range_image_to_point_cloud_labels(
    frame, range_images, segmentation_labels)
point_labels_ri2 = convert_range_image_to_point_cloud_labels(
    frame, range_images, segmentation_labels, ri_index=1)

# 3d points in vehicle frame.
points_all = np.concatenate(points, axis=0)
points_all_ri2 = np.concatenate(points_ri2, axis=0)
# point labels.
point_labels_all = np.concatenate(point_labels, axis=0)
point_labels_all_ri2 = np.concatenate(point_labels_ri2, axis=0)
# camera projection corresponding to each point.
cp_points_all = np.concatenate(cp_points, axis=0)
cp_points_all_ri2 = np.concatenate(cp_points_ri2, axis=0)

"""###Show colored point cloud
Example of rendered point clouds (this tutorial does not have visualization capability).
"""

from IPython.display import Image, display
display(Image('/content/waymo-od/tutorial/3d_semseg_points.png'))

frame.laser_labels

frame.projected_lidar_labels

frame.camera_labels

frame.no_label_zones
