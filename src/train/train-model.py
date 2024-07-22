from tqdm import tqdm_notebook, tnrange
import tensorflow as tf
if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.layers import ELU
from tensorflow.python.ops import array_ops
from tensorflow.python.keras.layers import Conv2D, Lambda, Dense, Multiply, Add
from tensorflow.python.util import deprecation as deprecation
from scipy.ndimage import median_filter, maximum_filter, percentile_filter
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
import glob
sess = tf.Session()

import keras
from keras import backend as K
K.set_session(sess)
from keras.losses import binary_crossentropy

from time import sleep
from scipy.ndimage import median_filter
from skimage.transform import resize
import pandas as pd
import numpy as np
from random import shuffle
import pandas as pd

import os
import random
import itertools
from keras.regularizers import l1
import tqdm
import hickle as hkl

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sys
import os
module_path = os.path.abspath(os.path.join('../src/'))
print(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)

import src.model
import src.losses
import src.data_utils


from src.adabound import AdaBoundOptimizer
from src.stochastic_weight_averaging import StochasticWeightAveraging

min_all = [0.006576638437476157, 0.0162050812542916, 0.010040436408026246, 0.013351644159609368, 
               0.01965362020294499, 0.014229037918669413, 0.015289539940489814, 0.011993591210803388,
               0.008239871824216068, 0.006546120393682765, 0.0, 0.0, 0.0, -0.1409399364817101,
               -0.4973397113668104, -0.09731556326714398, -0.7193834232943873]
max_all = [0.2691233691920348, 0.3740291447318227, 0.5171435111009385, 0.6027466239414053,
               0.5650263218127718, 0.5747005416952773, 0.5933928435187305, 0.6034943160143434, 
               0.7472037842374304, 0.7000076295109483, 0.509269855802243, 0.948334642387533,
               0.6729257769285485, 0.8177635298774327, 0.35768999002433816, 0.7545951919107605, 0.7602693339366691]


args = {
  'zoneout': 0.75,
  'base_filters': 64,
  'in_size': 28,
  'out_size': 14,
  'batch_size': 32,
  'init_lr': 8e-4,
  'length': 4,
  'n_bands': 17,
  'cosine_divider': 75,
  'warm_up_steps': 5000,
  'n_epochs': 100,
  'resume': True,
  'checkpoint_dir': 'ard_training/2024-07-19/',
  'train_data_folder': 'analysis_ready/',
  'test_data_folder': './',
  'mins': min_all,
  'maxs': max_all
}


print(f"Checkpoint directory: {args['checkpoint_dir']} \n"
      f"Training folder: {args['train_data_folder']} \n"
      f"Testing folder: {args['test_data_folder']} \n"
      f"Base filters: {args['base_filters']} \n"
      f"Resume: {args['resume']} \n"
      f"Input size: {args['in_size']} \n"
      f"Batch size: {args['batch_size']} \n"
      f"Num epochs: {args['n_epochs']} \n"
      )

def grad_norm(gradients):
  # For sharpness aware minimization
    norm = tf.compat.v1.norm(
      tf.stack([
          tf.compat.v1.norm(grad) for grad in gradients if grad is not None
      ])
    )
    return norm


def find_and_make_dirs(dirs: list) -> None:
    if not os.path.exists(os.path.realpath(dirs)):
        os.makedirs(os.path.realpath(dirs))


if __name__ == '__main__':
    if not os.path.exists(args['checkpoint_dir']):
        print(f"Making {args['checkpoint_dir']}")
        find_and_make_dirs(args['checkpoint_dir'])
    else:
        print(f"Already exists: {args['checkpoint_dir']}")
        
    # # Model definition
    reg = tf.contrib.layers.l2_regularizer(0.)
    temporal_model = True

    if temporal_model:
        inp = tf.placeholder(tf.float32, shape=(None, args['length'] + 1, args['in_size'], args['in_size'], args['n_bands']))
        length = tf.placeholder_with_default(np.full((1,), args['length'] + 1), shape = (None,))
    else:
        inp = tf.placeholder(tf.float32, shape=(None, args['in_size'], args['in_size'], args['n_bands']*5))
        
    labels = tf.placeholder(tf.float32, shape=(None, args['out_size'], args['out_size']))#, 1))
    keep_rate = tf.placeholder_with_default(1.0, ()) # For DropBlock
    is_training = tf.placeholder_with_default(False, (), 'is_training') # For DropBlock
    alpha = tf.placeholder(tf.float32, shape = ()) # For loss scheduling
    ft_lr = tf.placeholder_with_default(0.1, shape = ()) # For loss scheduling
    init_lr = tf.placeholder_with_default(0.0002, shape = ()) # For loss scheduling
    loss_weight = tf.placeholder_with_default(1.0, shape = ())
    beta_ = tf.placeholder_with_default(0.0, shape = ()) # For loss scheduling, not currently implemented
    keep_rate = tf.placeholder_with_default(1.0, ()) # For DropBlock

    #############
    # GRU BLOCK #
    #############
    gru_input = inp[:, :-1, ...]
    gru, steps = src.model.gru_block(inp = gru_input, length = length,
                                size = [args['in_size'], args['in_size'], ], # + 2 here for refleclt pad
                                flt = args['base_filters'] // 2,
                                scope = 'down_16',
                                train = is_training,
                                zoneout = args['zoneout'])
    with tf.variable_scope("gru_drop"):
        drop_block = src.model.DropBlock2D(keep_prob=keep_rate, block_size= 5)
        #mask = _mask(gru, is_training)
        #drop_block = DoDropBlock(keep_prob=keep_rate, block_size=5)
        gru = drop_block(gru, is_training)
        
    ###############
    # MEDIAN CONV #
    ###############
    median_input = inp[:, -1, ...]
    median_conv = src.model.conv_swish_gn(inp = median_input, is_training = is_training, stride = (1, 1),
                kernel_size = 3, scope = 'conv_median', filters = args['base_filters'], 
                keep_rate = keep_rate, activation = True, use_bias = False, norm = True,
                csse = True, dropblock = True, weight_decay = None, window_size = 15)

    ###########
    # ENCODER #
    ###########
    concat1 = tf.concat([gru, median_conv], axis = -1)
    #concat1 = median_conv
    concat = src.model.conv_swish_gn(inp = concat1, is_training = is_training, stride = (1, 1),
                kernel_size = 3, scope = 'conv_concat', filters = args['base_filters'],
                keep_rate = keep_rate, activation = True, use_bias = False, norm = True,
                csse = True, dropblock = True, weight_decay = None, padding = "SAME", window_size = 15)
    print(f"Concat: {concat.shape}")

        
    # MaxPool-conv-swish-GroupNorm-csse
    pool1 = MaxPool2D()(concat)
    conv1 = src.model.conv_swish_gn(inp = pool1, is_training = is_training, stride = (1, 1),
                kernel_size = 3, scope = 'conv1', filters = args['base_filters'] * 2,
                keep_rate = keep_rate, activation = True, use_bias = False, norm = True, padding = "VALID",
                csse = True, dropblock = True, weight_decay = None, window_size = 7)
    print(f"Conv1: {conv1.shape}")

    # MaxPool-conv-swish-csse-DropBlock
    pool2 = MaxPool2D()(conv1)
    conv2 = src.model.conv_swish_gn(inp = pool2, is_training = is_training, stride = (1, 1),
                kernel_size = 3, scope = 'conv2', filters = args['base_filters'] * 4, 
                keep_rate = keep_rate, activation = True,  use_bias = False, norm = True,
                csse = True, dropblock = True, weight_decay = None, block_size = 4, padding = "VALID",
                         window_size = 1)
    print("Encoded", conv2.shape)

    ###########
    # DECODER #
    ###########
    # Decoder 4 - 8, upsample-conv-swish-csse-concat-conv-swish
    up2 = tf.keras.layers.UpSampling2D((2, 2), interpolation = 'nearest')(conv2)
    up2 = src.model.conv_swish_gn(inp = up2, is_training = is_training, stride = (1, 1),
                        kernel_size = 3, scope = 'up2', filters = args['base_filters'] * 2, 
                        keep_rate = keep_rate, activation = True, use_bias = False, norm = True,
                        csse = True, dropblock = True, weight_decay = None, window_size = 3)
    conv1_crop = Cropping2D(2)(conv1)

    up2 = tf.concat([up2, conv1_crop], -1)
    up2 = src.model.conv_swish_gn(inp = up2, is_training = is_training, stride = (1, 1),
                        kernel_size = 3, scope = 'up2_out', filters = args['base_filters'] * 2, 
                        keep_rate =  keep_rate, activation = True,  use_bias = False, norm = True,
                        csse = True, dropblock = True, weight_decay = None, window_size = 3)

    # Decoder 8 - 14 upsample-conv-swish-csse-concat-conv-swish
    up3 = tf.keras.layers.UpSampling2D((2, 2), interpolation = 'nearest')(up2)
    up3 = src.model.conv_swish_gn(inp = up3, is_training = is_training, stride = (1, 1),
                        kernel_size = 3, scope = 'up3', filters = args['base_filters'], 
                        keep_rate = keep_rate, activation = True,  use_bias = False, norm = True,
                        csse = True, dropblock = True, weight_decay = None)
    gru_crop = Cropping2D(6)(concat)
    up3 = tf.concat([up3, gru_crop], -1)

    up3 = src.model.conv_swish_gn(inp = up3, is_training = is_training, stride = (1, 1),
                        kernel_size = 3, scope = 'out', filters = args['base_filters'], 
                        keep_rate  = keep_rate, activation = True,  use_bias = False, norm = True,
                        csse = True, dropblock = False, weight_decay = None, padding = "VALID")


    #print("Initializing last sigmoid bias with -2.94 constant")
    init = tf.constant_initializer([-np.log(0.68/0.32)]) # For focal loss
    print(f"The output is {up2.shape}, with a receptive field of {1}")
    fm = Conv2D(filters = 1,
                kernel_size = (1, 1),
                padding = 'valid',
                activation = 'sigmoid',
                bias_initializer = init,
               )(up3) # For focal loss

    ###################################################
    ######## MODEL IS DONE INITIALIZING BY HERE #######
    ###################################################

    print(f"The output, sigmoid is {fm.shape}")
    src.model.print_trainable_params()
    print(f"Starting model with: \n {args['zoneout']} zone out \n"
      f"{args['init_lr']} initial LR \n")  

    train_loss = src.losses.lovasz_surf(tf.reshape(labels, (-1, args['out_size'], args['out_size'], 1)), 
                             fm, weight = loss_weight, 
                             alpha = alpha, beta = beta_)

    test_loss = src.losses.lovasz_surf(tf.reshape(labels, (-1, args['out_size'], args['out_size'], 1)),
                            fm, weight = loss_weight, 
                            alpha = alpha, beta = beta_)
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    with tf.control_dependencies(update_ops):
        optimizer = AdaBoundOptimizer(init_lr, ft_lr, weight_decay = 2e-5) 
        ft_optimizer = tf.train.MomentumOptimizer(ft_lr, momentum = 0.8, use_nesterov = True)
        train_op = optimizer.minimize(train_loss)#, var_list = finetune_vars)   
        ft_op = ft_optimizer.minimize(train_loss)#, var_list = finetune_vars)

        # The following code blocks are for sharpness aware minimization
        # Adapted from https://github.com/sayakpaul/Sharpness-Aware-Minimization-TensorFlow
        # For tensorflow 1.15
        trainable_params = tf.trainable_variables()
        gradients = optimizer.compute_gradients(loss=train_loss, var_list=None)
        gradient_norm = grad_norm(gradients)
        scale = 0.05 / (gradient_norm + 1e-12)
        e_ws = []
        for (grad, param) in gradients:
            e_w = grad * scale
            param.assign_add(e_w)
            e_ws.append(e_w)

        sam_gradients = optimizer.compute_gradients(loss=train_loss, var_list=None)
        for (param, e_w) in zip(trainable_params, e_ws):
            param.assign_sub(e_w)
        train_step = optimizer.apply_gradients(sam_gradients)#, global_step = gstep)
        
        gradients2 = ft_optimizer.compute_gradients(loss=train_loss, var_list=None)
        gradient_norm2 = grad_norm(gradients2)
        scale2 = 0.05 / (gradient_norm2 + 1e-12)
        e_ws2 = []
        for (grad, param) in gradients2:
            e_w2 = grad * scale
            param.assign_add(e_w2)
            e_ws2.append(e_w2)

        sam_gradients2 = ft_optimizer.compute_gradients(loss=train_loss, var_list=None)
        for (param, e_w) in zip(trainable_params, e_ws2):
            param.assign_sub(e_w)
        ft_step = ft_optimizer.apply_gradients(sam_gradients2)
    
    # Create a saver to save the model each epoch
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    saver = tf.train.Saver(max_to_keep = 125)

    
    all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    all_vars = [x for x in all_vars if 'Momentum' not in x.name]
    all_vars = [x for x in all_vars if 'Backup' not in x.name]
    all_vars = [x for x in all_vars if 'StochasticWeight' not in x.name]
    all_vars = [x for x in all_vars if 'is_training' not in x.name]
    all_vars = [x for x in all_vars if 'n_models' not in x.name]
    all_vars = [x for x in all_vars if 'BackupVariables' not in x.name]
    all_vars = [x for x in all_vars if 'n_models' not in x.name]
    all_vars = [x for x in all_vars if 'AdaBelief' not in x.name]
    all_vars = [x for x in all_vars if 'RestoreV2' not in x.name]
        
    saver = tf.train.Saver(max_to_keep = 150, var_list = all_vars)

    if not os.path.exists(args['checkpoint_dir']):
        os.makedirs(args['checkpoint_dir'])
    if os.path.isfile(f"{args['checkpoint_dir']}metrics.npy"):
        metrics = np.load(f"{args['checkpoint_dir']}metrics.npy")
        print(f"Loading {args['checkpoint_dir']}metrics.npy")
    else:
        print("Starting anew")
        metrics = np.zeros((6, 300))

    if args['resume']:
        if len(os.listdir(args['checkpoint_dir'])) > 0:
            #checkpoints = [args['checkpoint_dir'] + x for x in os.listdir(args['checkpoint_dir'])]
            checkpoint = max(glob.glob(os.path.join(args['checkpoint_dir'], '*/')), key=os.path.getmtime)
            print(f'Resuming the latest checkpoint: {checkpoint} in {args["checkpoint_dir"]}')
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint))

    # SWA BLOCKS
    model_vars = tf.trainable_variables()
    swa = StochasticWeightAveraging()
    swa_op = swa.apply(var_list=model_vars)
    with tf.variable_scope('BackupVariables'):
        # force tensorflow to keep theese new variables on the CPU ! 
        backup_vars = [tf.get_variable(var.op.name, dtype=var.value().dtype, trainable=False,
                                 initializer=var.initialized_value())
                 for var in model_vars]

    # operation to assign SWA weights to model
    swa_to_weights = tf.group(*(tf.assign(var, swa.average(var).read_value()) for var in model_vars))
    # operation to store model into backup variables
    save_weight_backups = tf.group(*(tf.assign(bck, var.read_value()) for var, bck in zip(model_vars, backup_vars)))
    # operation to get back values from backup variables to model
    restore_weight_backups = tf.group(*(tf.assign(var, bck.read_value()) for var, bck in zip(model_vars, backup_vars)))
    src.model.initialize_uninitialized(sess)

    #####################################################
    ######## DEALING WITH THE TRAINING DATA BELOW #######
    #####################################################


    

    normalize = False
    # The train X has already had:
    # - the radar bands converted to decibel
    # - the indices added
    # - the quarterly medians calculated
    # - has been normalized
    train_x = np.load(f"{args['train_data_folder']}/original_x.npy")
    train_y = np.load(f"{args['train_data_folder']}/original_y.npy")
    #data = pd.read_csv(f"{args['train_data_folder']}train_x.csv")
    
    
    # The test X has already had:
    # - the radar bands converted to decibel
    # - the indices added
    # - the quarterly medians calculated
    # - but has not been normalized
    test_x = np.load(f"{args['test_data_folder']}/test_x.npy")
    test_y = np.load(f"{args['test_data_folder']}/test_y.npy")
    #test_data = pd.read_csv(f"{args['test_data_folder']}/test_x.csv")

    for band in range(0, test_x.shape[-1]):
        mins = min_all[band]
        maxs = max_all[band]
        test_x[..., band] = np.clip(test_x[..., band], mins, maxs)
        midrange = (maxs + mins) / 2
        rng = maxs - mins
        standardized = (test_x[..., band] - midrange) / (rng / 2)
        test_x[..., band] = standardized    
        
    train_ids = [x for x in range(len(train_y))]
    batch = src.data_utils.equibatch(train_ids, train_y)
    x_batch_test, y_batch_test = src.data_utils.augment_batch([x for x in range(32)], 32, train_x, train_y, args)

    fine_tune = False
    ft_epochs = 0
    f1 = 0
    best_val = 0
    for i in range(1, args['n_epochs']):
        print(f'Starting epoch {i}')
        max_keep_rate = 0.5
        SWA = False
        fine_tune = False
        if i > args['n_epochs'] - 15:
            SWA = True
            fine_tune = True
            ft_epochs += 1

        ft_learning_rate = 0.1
        
        if i >= 3:
            max_keep_rate = 0.40
        if i >= 15:
            max_keep_rate = np.minimum(0.45, max_keep_rate)
        if i >= 40:
            max_keep_rate = i * 0.01
            max_keep_rate = np.minimum(0.45, max_keep_rate)

        al = np.min( [0.01 * (i - 1), 0.2] )
        be = 0.0
        test_al = al
        if fine_tune == True:
            op = ft_op
            print(f"FINE TUNING WITH {ft_learning_rate} LR")
        else:
            op = train_step
            
        train_ids = [x for x in range(len(train_y))]
        randomize = train_ids
        randomize = src.data_utils.equibatch(train_ids, train_y)
        loss = train_loss
        cosine_divider = args['cosine_divider']
        cosine_epoch = i % cosine_divider

        test_ids = [x for x in range(0, len(test_x))]
        losses = []
        
        keeprate = np.max(((1.025 - (cosine_epoch * 0.025)
                                   ), max_keep_rate))

        warm_up_steps = (i - 1) * (len(train_y) // args['batch_size'])
        
        if i < 200:           
            cosdec = src.losses.calc_cosine_decay(cosine_epoch, cosine_divider, 0)
        adam_lr = args['init_lr'] * cosdec
        ft_learning_rate = args['init_lr'] * 100 * cosdec
        
        print(f"starting epoch {i}, alpha: {al}, beta: {be} drop: {keeprate}"
             f" Learning rate: {ft_learning_rate}, {adam_lr}")
        
        n_batch = int(len(randomize) // args['batch_size'])
        n_step = 0
        for k in tqdm.trange(int(len(randomize) // (3 * args['batch_size']))):
            batch_ids = randomize[k*args['batch_size']:(k+1)*args['batch_size']]
            x_batch, y_batch = src.data_utils.augment_batch(batch_ids, args['batch_size'], train_x, train_y, args)
            warm_up_steps += 1
            n_step += 1
            cosine_epoch = cosine_epoch + (1 / n_batch)
            cosdec = src.losses.calc_cosine_decay(cosine_epoch, cosine_divider, 0)
            adam_lr = args['init_lr'] * cosdec
            ft_learning_rate = 1e-1 * cosdec
            if warm_up_steps < args['warm_up_steps']:
                adam_lr = ((warm_up_steps) / args['warm_up_steps']) * 2e-4
                ft_learning_rate= ((warm_up_steps) / args['warm_up_steps']) * 2e-2
            opt, tr = sess.run([train_step, loss], # op needs to be train_step for SAM
                              feed_dict={inp: x_batch,
                                         length: np.full((args['batch_size'],), args['length']),
                                         labels: y_batch,
                                         is_training: True,
                                         loss_weight: 1.0,
                                         keep_rate: keeprate,
                                         alpha: al,
                                         beta_: be,
                                         init_lr: adam_lr,
                                         ft_lr: ft_learning_rate,
                                         })
            losses.append(tr)  

        print(f"Epoch {i}: Loss {np.around(np.mean(losses[:-1]), 3)}")
        metrics[0, i] = np.mean(losses[:-1])
        run_metrics = True
        if run_metrics:
            if SWA:
                sess.run(save_weight_backups)
                sess.run(swa_to_weights)
            val_loss, f1, error, haus, dice = src.losses.calculate_metrics(test_x = test_x, 
                                                                           test_y = test_y,
                                                                           sess = sess,
                                                                           op = fm,
                                                                           args = args,
                                                                           input_ops = [inp, length, is_training, labels, loss_weight, alpha],
                                                                           test_loss = test_loss,
                                                                           al = test_al, 
                                                                           canopy_thresh = 75, 
                                                                           )
            metrics[1, i] = val_loss
            metrics[2, i] = error
            metrics[3, i] = haus
            metrics[4, i] = dice
            metrics[5, i] = f1
            if SWA:
                sess.run(swa_op)

            if f1 > (best_val - 0.02):
                print(f"Saving model with {f1}")
                np.save(f"{args['checkpoint_dir']}/metrics.npy", metrics)
                os.mkdir(f"{args['checkpoint_dir']}/{str(i)}-{str(f1*100)[:2]}-{str(f1*100)[3]}/")
                save_path = saver.save(sess, f"{args['checkpoint_dir']}/{str(i)}-{str(f1*100)[:2]}-{str(f1*100)[3]}/model")
                if f1 > best_val:
                    best_val = f1
            if SWA:
                sess.run(restore_weight_backups)

    save_path = saver.save(sess, f"{args['checkpoint_dir']}/{str(i)}-{str(f1*100)[:2]}-{str(f1*100)[3]}/model")