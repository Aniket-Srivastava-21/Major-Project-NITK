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


from keras.models import *
from keras.layers import *
from keras.optimizers import *

def unet(pretrained_weights = None, input_size = (256,256,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(25, 1, activation = 'softmax')(conv9)

    model = Model(inputs = inputs, outputs = conv10)
    
    model.summary()

    return model
X_train_scaled.shape

X_train_resized = np.resize(X_train_scaled, (24, 896, 1920, 3))
X_test_resized = np.resize(X_test_scaled, (6, 896, 1920, 3))

y_train_resized = np.resize(y_train_cat, (24, 896, 1920, 25))
y_test_resized = np.resize(y_test_cat, (6, 896, 1920, 25))

model = unet(input_size = (896, 1920, 3))

model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])

X_train_resized.shape, X_test_resized.shape

np.save('/content/drive/MyDrive/Major project/X_train_final.npy', X_train_resized)
np.save('/content/drive/MyDrive/Major project/X_test_final.npy', X_test_resized)
np.save('/content/drive/MyDrive/Major project/y_train_final.npy', y_train_resized)
np.save('/content/drive/MyDrive/Major project/y_test_final.npy', y_test_resized)

X_train_resized = np.load('/content/drive/MyDrive/Major project/X_train_final.npy')
X_test_resized = np.load('/content/drive/MyDrive/Major project/X_test_final.npy')
y_train_resized = np.load('/content/drive/MyDrive/Major project/y_train_final.npy')
y_test_resized = np.load('/content/drive/MyDrive/Major project/y_test_final.npy')

history = model.fit(X_train_resized, y_train_resized, epochs = 20, batch_size = 4, validation_data = (X_test_resized, y_test_resized))
