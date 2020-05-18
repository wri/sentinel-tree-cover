inp_first_half = inp[:, :, :, :, :10]
inp_second_half = inp[:, :, :, :, 11:]
no_dem = tf.concat([inp_first_half, inp_second_half], axis = -1)
dem = tf.reshape(tf.reduce_mean(inp[:, :, :, :, 10], axis = 1), (-1, 16, 16, 1))
gru_out, steps = gru_block(inp = no_dem, length = length2, 
                            size = [16, 16], 
                            flt = gru_flt, 
                            scope = 'down_16', 
                            train = is_training)

#steps_a2 = a2_block(steps, gru_flt*2, gru_flt//2, gru_flt//2, k = 1)
#print("Attention shape: {}".format(steps_a2.shape))
gru_out = tf.concat([gru_out, dem], axis = -1)
csse1 = csse_block(gru_out, 'csse1')
drop_block1 = DropBlock2D(keep_prob=keep_rate, block_size=4)
csse1 = drop_block1(csse1, is_training)

# Light FPA, CSSE, 4x4 Drop block
fpa1 = fpa(csse1, is_training, fpa_flt)
csse2 = csse_block(fpa1, 'csse2')
drop_block2 = DropBlock2D(keep_prob=keep_rate, block_size=3)
csse2 = drop_block2(csse2, is_training)


# Skip connect
x = tf.concat([csse2, csse1], axis = -1)
drop_block3 = DropBlock2D(keep_prob=keep_rate, block_size=2)
x = drop_block3(x, is_training)

x = conv_bn_elu(x, is_training, 3, "out_2", out_conv_flt, False, 'valid')
drop_block4 = DropBlock2D(keep_prob=keep_rate, block_size=1)
x = drop_block4(x, is_training)

print("Initializing last sigmoid bias with -2.94 constant")
init = tf.constant_initializer([-np.log(0.7/0.3)]) # For focal loss
fm = Conv2D(filters = 1,
            kernel_size = (1, 1), 
            padding = 'valid',
            activation = 'sigmoid',
            bias_initializer = init,
           )(x) # For focal loss