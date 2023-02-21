"""# Using Saved Data to train Model"""

image_arr = np.load('image_array.npy')
label_arr = np.load('label_array.npy')

image_arr.shape, label_arr.shape

plt.imshow(label_arr[0][:,:,0])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(image_arr[:30], label_arr[:30], test_size = 0.2, random_state = 1)

X_train.shape, X_test.shape

category_map = {}
values = [  0,   3,   7,   8,  11,  15,  19,  27,  35,  50,  54,  58,  62,
        66,  70,  74,  78,  82,  85,  89,  93,  97, 101, 105, 109]

for i in range(25):
  category_map[values[i]] = i

category_map

for key, val in category_map.items():
  y_train[y_train==key]=val

np.unique(y_train)

for key, val in category_map.items():
  y_test[y_test==key]=val

np.unique(y_test)

from keras.utils import to_categorical

y_train_cat = to_categorical(y_train, num_classes=25, dtype = 'bool')

y_train_cat.shape

y_test_cat = to_categorical(y_test, num_classes=25, dtype = 'bool')
y_test_cat.shape

X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0

X_train.shape, X_test.shape

X_train_scaled.shape

X_train_resized = np.resize(X_train_scaled, (24, 896, 1920, 3))
X_test_resized = np.resize(X_test_scaled, (6, 896, 1920, 3))

y_train_resized = np.resize(y_train_cat, (24, 896, 1920, 25))
y_test_resized = np.resize(y_test_cat, (6, 896, 1920, 25))

X_train_resized.shape, X_test_resized.shape

np.save('/content/drive/MyDrive/Major project/X_train_final.npy', X_train_resized)
np.save('/content/drive/MyDrive/Major project/X_test_final.npy', X_test_resized)
np.save('/content/drive/MyDrive/Major project/y_train_final.npy', y_train_resized)
np.save('/content/drive/MyDrive/Major project/y_test_final.npy', y_test_resized)
