import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools
import sys
import matplotlib.pyplot as plt

tf.enable_eager_execution()

file = tarfile.open('large_dataset.tar')
  
# extracting file
file.extractall('./Combined_dataset')

file.close()

os.remove('large_dataset.tar')

directory = 'Combined_dataset'

label_list = []
image_list = []

done_count = 0

for filename in os.listdir(directory):
  FILENAME = os.path.join(directory, filename)

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

  print(FILENAME, "Done!")
  print(len(label_list), len(image_list))

len(label_list), len(image_list)

plt.imshow(image_list[0])

plt.imshow(label_list[0][:,:,0])

sys.getsizeof(image_list[0])

5103496 * 394

sys.getsizeof(label_list[0])

image_list[1].shape

img_arr = np.array(image_list)

label_arr = np.array(label_list)

sys.getsizeof(img_arr), sys.getsizeof(label_arr)

np.save('image_array.npy', img_arr)

np.save('label_array.npy', label_arr)
