import tensorflow as tf

def Xception(images, num_classes):

    regularizer = tf.keras.regularizers.l2()

    with tf.name_scope('base'):
        layer = tf.keras.layers.Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularizer, use_bias=False)(
            images)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Activation('relu')(layer)
        layer = tf.keras.layers.Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularizer, use_bias=False)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Activation('relu')(layer)

    with tf.name_scope('module1'):
        residual = tf.keras.layers.Conv2D(16, (1, 1), strides=(2, 2), padding='same', use_bias=False)(layer)
        residual = tf.keras.layers.BatchNormalization()(residual)
        layer = tf.keras.layers.SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularizer,
                                                use_bias=False)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Activation('relu')(layer)
        layer = tf.keras.layers.SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularizer,
                                                use_bias=False)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(layer)
        layer = tf.keras.layers.add([layer, residual])

    with tf.name_scope('module2'):
        residual = tf.keras.layers.Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(layer)
        residual = tf.keras.layers.BatchNormalization()(residual)
        layer = tf.keras.layers.SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularizer,
                                                use_bias=False)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Activation('relu')(layer)
        layer = tf.keras.layers.SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularizer,
                                                use_bias=False)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(layer)
        layer = tf.keras.layers.add([layer, residual])

    with tf.name_scope('module3'):
        residual = tf.keras.layers.Conv2D(64, (1, 1), strides=(2, 2), padding='same', use_bias=False)(layer)
        residual = tf.keras.layers.BatchNormalization()(residual)
        layer = tf.keras.layers.SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=regularizer,
                                                use_bias=False)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Activation('relu')(layer)
        layer = tf.keras.layers.SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=regularizer,
                                                use_bias=False)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(layer)
        layer = tf.keras.layers.add([layer, residual])

    with tf.name_scope('module4'):
        residual = tf.keras.layers.Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(layer)
        residual = tf.keras.layers.BatchNormalization()(residual)
        layer = tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same', kernel_regularizer=regularizer,
                                                use_bias=False)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Activation('relu')(layer)
        layer = tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same', kernel_regularizer=regularizer,
                                                use_bias=False)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(layer)
        layer = tf.keras.layers.add([layer, residual])

    with tf.name_scope('output'):
        layer = tf.keras.layers.Conv2D(num_classes, (3, 3), padding='same')(layer)
        layer = tf.keras.layers.GlobalAveragePooling2D()(layer)
        output = tf.keras.layers.Activation('softmax')(layer)

    return output

