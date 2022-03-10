print(inp_pad.shape)

with tf.variable_scope('10'):
    # Downsampling Block 1 (14 x 14)
    cell_10 = ConvGRUCell(shape = [16, 16],
                   filters = 8,
                   kernel = [3,3],
                   padding = 'SAME')

# Return the final state and the output states
first_conv = convGRU(inp_pad, cell_10, length2)
print("FIRST GRU {}".format(first_conv[0].shape))

downsampled = TimeDistributed(MaxPool2D(pool_size = (2, 2)))(first_conv[0])
print("DOWNSAMPLE {}".format(downsampled.shape))

# Downsampling block 2 (7 x 7)
with tf.variable_scope('8'):
    cell_7 = ConvGRUCell(shape = [8, 8],
                   filters = 16,
                   kernel = [3,3],
                   padding = 'SAME')
    state_7 = convGRU(downsampled, cell_7, length2)
downsampled_4 = TimeDistributed(MaxPool2D(pool_size = (2, 2)))(state_7[0])
print("SECOND GRU {}".format(state_7[1].shape))

with tf.variable_scope('4'):
    cell_4 = ConvGRUCell(shape = [4, 4],
                   filters = 32,
                   kernel = [3,3],
                   padding = 'SAME')
    state_4 = convGRU(downsampled_4, cell_4, length2)
print("THIRD GRU {}".format(state_4[1].shape))

# 4x4 - 4x4
conv_block_7_u = Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same', activity_regularizer=reg)(state_4[1])
elu_7_u = ELU()(conv_block_7_u)
x = Batch_Normalization(elu_7_u, training=is_training, scope = 'bn1')
#x = batchnorm(elu_7_u, is_training, 'bn1')
squeezed = Squeeze_excitation_layer(input_x = x, out_dim = 32, ratio = 4, layer_name = "squeezed")
print("Down block conv {}".format(elu_7_u.shape))

# 4x4 - 8x8
upsampling_8 = tf.keras.layers.Conv2DTranspose(filters = 24, kernel_size = (3, 3), strides=(2, 2), padding='same')(squeezed)
concat_8 = Concatenate(axis = -1)([upsampling_8, state_7[1]])
padded_8 = ReflectionPadding2D((1, 1))(concat_8)
conv_8 = Conv2D(filters = 24,
            kernel_size = (3, 3),
            padding = 'valid',
            activity_regularizer=reg,
            )(padded_8)
elu_8 = ELU()(conv_8)
#bn_8 = batchnorm(elu_8, is_training, 'bn2')
bn_8 = Batch_Normalization(elu_8, training=is_training, scope = 'bn8')
squeeze_8 = Squeeze_excitation_layer(input_x = bn_8, out_dim = 24, ratio = 4, layer_name = "squeezed_8")
print("Upblock 8 {}".format(squeeze_8.shape))

# 8x8 - 16 x 16
upsampling_16 = tf.keras.layers.Conv2DTranspose(filters = 16, kernel_size = (3, 3), strides=(2, 2), padding='same')(squeeze_8)
concat_16 = Concatenate(axis = -1)([upsampling_16, first_conv[1]])
padded_16 = ReflectionPadding2D((1, 1))(concat_16)
conv_16 = Conv2D(filters = 16,
            kernel_size = (3, 3),
            padding = 'valid',
            activity_regularizer=reg,
            )(padded_16)
elu_16 = ELU()(conv_16)
#bn_16 = batchnorm(elu_16, is_training, 'bn3')
bn_16 = Batch_Normalization(elu_16, training=is_training, scope = 'bn16')
squeezed_16 = Squeeze_excitation_layer(input_x = bn_16, out_dim = 16, ratio = 4, layer_name = "squeezed_16")
print("Up block 16 {}".format(squeezed_16.shape))

#padded = ReflectionPadding2D((1, 1))(squeezed_16)
#fm = Conv2D(filters = 12,
#            kernel_size = (3, 3),
#            padding = 'valid',
#            activity_regularizer=reg,
#            )(padded)
#elu = ELU()(fm)
#bn_final = Batch_Normalization(elu, training=is_training, scope = 'bnfinal')
#squeezed_16 = Squeeze_excitation_layer(input_x = bn_final, out_dim = 12, ratio = 4, layer_name = "squeezed_final")
#print("Up block conv 3 {}".format(squeezed_16.shape))
# Output layer
fm = Conv2D(filters = 1,
            kernel_size = (1, 1),
            padding = 'valid',
            activation = 'sigmoid'
            )(squeezed_16)
print(fm.shape)
