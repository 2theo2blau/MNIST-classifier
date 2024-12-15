import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Add
from keras.utils import to_categorical
from keras.optimizers import Adam

# input image dimensions
img_rows, img_cols = 28, 28
num_classes = 10

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the image data into a 4D array with dimensions (batch, height, width, channels)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1).astype('float32') / 255

# Convert class vectors to binary class matrices (one-hot encoding)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

def resnet_block(x, num_filters):
    f1 = Conv2D(num_filters, (3, 3), padding='same')(x)
    f1 = BatchNormalization()(f1)
    f1 = Activation('relu')(f1)
    f1 = Conv2D(num_filters, (3, 3), padding='same')(f1)
    return Add()([x, f1])  # skip connection

def wide_block(x, num_filters):
    f1 = Conv2D(num_filters, (1, 1), padding='same')(x)
    f1 = BatchNormalization()(f1)
    f1 = Activation('relu')(f1)
    
    f2 = Conv2D(num_filters, (3, 3), padding='same')(f1)
    f2 = BatchNormalization()(f2)
    f2 = Activation('relu')(f2)
    
    f3 = Conv2D(num_filters * 4, (1, 1), padding='same')(f2)
    return Add()([x, f3])  # skip connection

# def create_deeper_cnn(input_shape):
#     model = Sequential()
#     model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(BatchNormalization())
#     model.add(Conv2D(64, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(BatchNormalization())
#     model.add(Conv2D(128, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#     model.add(Flatten())
#     model.add(Dense(256, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(num_classes, activation='softmax'))
#     return model

def create_wide_resnet(input_shape):
    input_tensor = keras.Input(shape=input_shape)
    x = input_tensor
    
    # Initial convolution block
    x = BatchNormalization()(x)
    f1 = Conv2D(16, (3, 3), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(f1)
    
    # Depth layers
    for i in range(4):  # You can adjust the number of blocks here
        x = wide_block(x, 16 * (i + 1))
        
    # Global average pooling and classification layer
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=input_tensor, outputs=x)

# Choose one of the architectures
model = create_deeper_cnn((img_rows, img_cols, 1))
# model = create_wide_resnet((img_rows, img_cols, 1))

# compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# train the model
model.fit(x_train, y_train, batch_size=64, epochs=20, verbose=1, validation_data=(x_test, y_test))

# evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
