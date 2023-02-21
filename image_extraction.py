import os
import tensorflow.compat.v1 as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
import numpy as np
import sys
import matplotlib.pyplot as plt

tf.enable_eager_execution()

FILENAME = 'dataset.tfrecord'

label_list = []
image_list = []

dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')

try:
  for data in dataset:
      frame = open_dataset.Frame()
      frame.ParseFromString(bytearray(data.numpy()))

      for i in range(len(frame.images)):
          panoptic_image = frame.images[i].camera_segmentation_label.panoptic_label
      try:
          label = tf.image.decode_png(panoptic_image)
          img = tf.image.decode_jpeg(frame.images[i].image)
          label_list.append(label.numpy())
          image_list.append(img.numpy())
      except:
          pass

  done_count += 1

except Exception as e:
    print("Error:", e)

print("Done!")
print(len(label_list), len(image_list))

len(label_list), len(image_list)

plt.imshow(image_list[0])

plt.imshow(label_list[0][:,:,0])

sys.getsizeof(image_list[0])

sys.getsizeof(label_list[0])

image_list[1].shape

img_arr = np.array(image_list)

label_arr = np.array(label_list)

sys.getsizeof(img_arr), sys.getsizeof(label_arr)

np.save('image_array.npy', img_arr)

np.save('label_array.npy', label_arr)
