# ==============================
# CNN MODEL ARCHITECTURE
# ==============================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dropout,
    Dense,
    Flatten
)

model = Sequential()

# --------------------------------
# BLOCK 1
# --------------------------------

model.add(
    Conv2D(
        filters=32,
        kernel_size=(3,3),
        activation='relu',
        padding='same',
        input_shape=(224,224,3)
    )
)

model.add(BatchNormalization())

model.add(
    Conv2D(
        filters=32,
        kernel_size=(3,3),
        activation='relu',
        padding='same'
    )
)

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))


# --------------------------------
# BLOCK 2
# --------------------------------

model.add(
    Conv2D(
        filters=64,
        kernel_size=(3,3),
        activation='relu',
        padding='same'
    )
)

model.add(BatchNormalization())

model.add(
    Conv2D(
        filters=64,
        kernel_size=(3,3),
        activation='relu',
        padding='same'
    )
)

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))


# --------------------------------
# BLOCK 3
# --------------------------------

model.add(
    Conv2D(
        filters=128,
        kernel_size=(3,3),
        activation='relu',
        padding='same'
    )
)

model.add(BatchNormalization())

model.add(
    Conv2D(
        filters=128,
        kernel_size=(3,3),
        activation='relu',
        padding='same'
    )
)

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.30))


# --------------------------------
# BLOCK 4
# --------------------------------

model.add(
    Conv2D(
        filters=256,
        kernel_size=(3,3),
        activation='relu',
        padding='same'
    )
)

model.add(BatchNormalization())

model.add(
    Conv2D(
        filters=256,
        kernel_size=(3,3),
        activation='relu',
        padding='same'
    )
)

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.35))


# --------------------------------
# BLOCK 5
# --------------------------------

model.add(
    Conv2D(
        filters=512,
        kernel_size=(3,3),
        activation='relu',
        padding='same'
    )
)

model.add(BatchNormalization())

model.add(
    Conv2D(
        filters=512,
        kernel_size=(3,3),
        activation='relu',
        padding='same'
    )
)

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.40))


# --------------------------------
# FLATTEN
# --------------------------------

model.add(Flatten())


# --------------------------------
# DENSE LAYERS
# --------------------------------

model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.50))

model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.50))

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.40))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.30))


# --------------------------------
# OUTPUT LAYER
# --------------------------------

model.add(
    Dense(
        num_classes,
        activation='softmax'
    )
)


# --------------------------------
# COMPILE MODEL
# --------------------------------

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()