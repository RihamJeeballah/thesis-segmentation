import tensorflow as tf
from tensorflow.keras import layers, Model

def conv_block(inputs, filters, name, dropout_rate=0.3):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name}_conv1")(inputs)
    x = layers.Dropout(dropout_rate, name=f"{name}_drop1")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name}_conv2")(x)
    x = layers.Dropout(dropout_rate, name=f"{name}_drop2")(x)
    return x

def encoder_block(inputs, filters, name, dropout_rate=0.3):
    x = conv_block(inputs, filters, name, dropout_rate=dropout_rate)
    p = layers.MaxPooling2D((2, 2), name=f"{name}_pool")(x)
    return x, p

def build_dual_encoder_with_shared_tail_adapted(input_shape=(512, 512, 1), filters=[32, 64, 128, 256, 512], dropout_rate=0.3):
    input_ct = layers.Input(shape=input_shape, name="CT_input")

    # Adaptation for Encoder A (MRI → CT)
    adapted_a = layers.Conv2D(1, 1, activation=None, name="adapt_conv")(input_ct)
    adapted_a = layers.BatchNormalization(name="adapt_bn")(adapted_a)

    # Encoder A
    xa0, pa0 = encoder_block(adapted_a, filters[0], "A_conv00", dropout_rate)
    xa1, pa1 = encoder_block(pa0, filters[1], "A_conv10", dropout_rate)
    xa2, pa2 = encoder_block(pa1, filters[2], "A_conv20", dropout_rate)
    xa3, pa3 = encoder_block(pa2, filters[3], "A_conv30", dropout_rate)
    xa4 = conv_block(pa3, filters[4], "A_conv40", dropout_rate)

    # Encoder B
    xb0, pb0 = encoder_block(input_ct, filters[0], "B_conv00", dropout_rate)
    xb1, pb1 = encoder_block(pb0, filters[1], "B_conv10", dropout_rate)
    xb2, pb2 = encoder_block(pb1, filters[2], "B_conv20", dropout_rate)
    xb3, pb3 = encoder_block(pb2, filters[3], "B_conv30", dropout_rate)
    xb4 = conv_block(pb3, filters[4], "B_conv40", dropout_rate)

    # Shared Tail
    merged = layers.Concatenate(name="shared_concat")([xa4, xb4])
    s1, ps1 = encoder_block(merged, filters[4], "shared_conv50", dropout_rate)
    s2 = conv_block(ps1, filters[4], "shared_conv60", dropout_rate)

    # Decoder
    u3 = layers.Conv2DTranspose(filters[3], 2, strides=2, padding="same", name="up3")(s2)
    u3 = layers.Concatenate(name="merge3")([u3, s1])
    x3 = conv_block(u3, filters[3], "dec_x30", dropout_rate)

    u2 = layers.Conv2DTranspose(filters[2], 2, strides=2, padding="same", name="up2")(x3)
    x2 = conv_block(u2, filters[2], "dec_x20", dropout_rate)

    u1 = layers.Conv2DTranspose(filters[1], 2, strides=2, padding="same", name="up1")(x2)
    x1 = conv_block(u1, filters[1], "dec_x10", dropout_rate)

    u0 = layers.Conv2DTranspose(filters[0], 2, strides=2, padding="same", name="up0")(x1)
    x0 = conv_block(u0, filters[0], "dec_x00", dropout_rate)

    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="output")(x0)

    return Model(inputs=input_ct, outputs=outputs, name="DualEncoderSharedTail_Adapted")
