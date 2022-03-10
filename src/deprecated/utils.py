import numpy as np
import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
import itertools
from tensorflow.python.keras.layers import Conv2D, Lambda, Dense, Multiply, Add

INPUT_FOLDER = "/".join(OUTPUT_FOLDER.split("/")[:-2]) + "/"
def process_multiple_years(coord: tuple,
                       step_x: int,
                       step_y: int,
                       path: str = INPUT_FOLDER) -> None:
    '''Wrapper function to interpolate clouds and temporal gaps, superresolve tiles,
       calculate relevant indices, and save analysis-ready data to the output folder

       Parameters:
        coord (tuple)
        step_x (int):
        step_y (int):
        folder (str):

       Returns:
        None
    '''

    idx = str(step_y) + "_" + str(step_x)
    x_vals, y_vals = make_folder_names(step_x, step_y)

    d2017 = hkl.load(f"{path}/2017/interim/dates_{idx}.hkl")
    d2018 = hkl.load(f"{path}/2018/interim/dates_{idx}.hkl")
    d2019 = hkl.load(f"{path}/2019/interim/dates_{idx}.hkl")

    x2017 = hkl.load(f"{path}/2017/interim/{idx}.hkl").astype(np.float32)
    x2018 = hkl.load(f"{path}/2018/interim/{idx}.hkl").astype(np.float32)
    x2019 = hkl.load(f"{path}/2019/interim/{idx}.hkl").astype(np.float32)

    s1_all = np.empty((72, 646, 646, 2))
    s1_2017 = hkl.load(f"{path}/2017/raw/s1/{idx}.hkl")
    s1_all[:24] = s1_2017
    s1_2018 = hkl.load(f"{path}2018/raw/s1/{idx}.hkl")
    s1_all[24:48] = s1_2018
    s1_2019 = hkl.load(f"{path}2019/raw/s1/{idx}.hkl")
    s1_all[48:] = s1_2019


    index = 0
    tiles = tile_window(IMSIZE, IMSIZE, window_size = 142)
    for t in tiles:
        start_x, start_y = t[0], t[1]
        end_x = start_x + t[2]
        end_y = start_y + t[3]
        s2017 = x2017[:, start_x:end_x, start_y:end_y, :]
        s2018 = x2018[:, start_x:end_x, start_y:end_y, :]
        s2019 = x2019[:, start_x:end_x, start_y:end_y, :]
        s2017, _  = calculate_and_save_best_images(s2017, d2017)
        s2018, _ = calculate_and_save_best_images(s2018, d2018)
        s2019, _ = calculate_and_save_best_images(s2019, d2019)
        subtile = np.empty((72*3, 142, 142, 15))
        subtile[:72] = s2017
        subtile[72:144] = s2018
        subtile[144:] = s2019
        print(np.sum(np.isnan(subtile), axis = (1, 2, 3)))
        out_17 = f"{path}/2017/processed/{y_vals[index]}/{x_vals[index]}.hkl"
        out_18 = f"{path}/2018/processed/{y_vals[index]}/{x_vals[index]}.hkl"
        out_19 = f"{path}/2019/processed/{y_vals[index]}/{x_vals[index]}.hkl"

        index += 1
        print(f"{index}: The output file is {out_17}")
        subtile = interpolate_array(subtile, dim = 142)
        subtile = np.concatenate([subtile, s1_all[:, start_x:end_x, start_y:end_y, :]], axis = -1)
        for folder in [out_17, out_18, out_19]:
            output_folder = "/".join(folder.split("/")[:-1])
            if not os.path.exists(os.path.realpath(output_folder)):
                os.makedirs(os.path.realpath(output_folder))
        subtile = to_int32(subtile)
        assert subtile.shape[1] == 142, f"subtile shape is {subtile.shape}"

        hkl.dump(subtile[:24], out_17, mode='w', compression='gzip')
        hkl.dump(subtile[24:48], out_18, mode='w', compression='gzip')
        hkl.dump(subtile[48:], out_19, mode='w', compression='gzip')

def reject_outliers(data, m = 4):
    d = data - np.median(data, axis = (0))
    mdev = np.median(data, axis = 0)
    s = d / mdev
    n_changed = 0
    for x in tnrange(data.shape[1]):
        for y in range(data.shape[2]):
            for band in range(data.shape[3]):
                to_correct = np.where(s[:, x, y, band] > m)
                data[to_correct, x, y, band] = mdev[x, y, band]
                n_changed += len(to_correct[0])
    print(f"Rejected {n_changed} outliers")
    return data


def convertCoords(xy, src='', targ=''):
    """ Converts coords from one EPSG to another

        Parameters:
         xy (tuple): input longitiude, latitude tuple
         src (str): EPSG code associated with xy
         targ (str): EPSG code of target output

        Returns:
         pt (tuple): (x, y) tuple of xy in targ EPSG
    """

    srcproj = osr.SpatialReference()
    srcproj.ImportFromEPSG(src)
    targproj = osr.SpatialReference()
    if isinstance(targ, str):
        targproj.ImportFromProj4(targ)
    else:
        targproj.ImportFromEPSG(targ)
    transform = osr.CoordinateTransformation(srcproj, targproj)

    pt = ogr.Geometry(ogr.wkbPoint)
    pt.AddPoint(xy[0], xy[1])
    pt.Transform(transform)
    return([pt.GetX(), pt.GetY()])

def ndvi(x, verbose = False):
    # (B8 - B4)/(B8 + B4)
    NIR = x[:, :, :, 6]
    RED = x[:, :, :, 2]
    ndvis = (NIR-RED) / (NIR+RED)
    if verbose:
        mins = np.min(ndvis)
        maxs = np.max(ndvis)
        if mins < -1 or maxs > 1:
            print("ndvis error: {}, {}".format(mins, maxs))
    x = np.concatenate([x, ndvis[:, :, :, np.newaxis]], axis = -1)
    return x

def evi(x, verbose = False):
    # 2.5 x (08 - 04) / (08 + 6 * 04 - 7.5 * 02 + 1)
    NIR = x[:, :, :, 6]
    RED = x[:, :, :, 2]
    BLUE = x[:, :, :, 0]
    evis = 2.5 * ( (NIR-RED) / (NIR + (6*RED) - (7.5*BLUE) + 1))
    if verbose:
        amin = np.argwhere(np.array([np.min(evis[i]) for i in range(x.shape[0])]) < -3)
        amax = np.argwhere(np.array([np.max(evis[i]) for i in range(x.shape[0])]) > 3)
        amin = np.concatenate([amin, amax])
        mins = np.min(evis)
        maxs = np.max(evis)
        if mins < -1 or maxs > 1:
            print("evis error: {}, {}, {} step, clipping to -1.5, 1.5".format(mins, maxs, amin))
    evis = np.clip(evis, -1.5, 1.5)
    x = np.concatenate([x, evis[:, :, :, np.newaxis]], axis = -1)
    return x, amin

def bounding_box(points, expansion = 160):
    """ Calculates the corners of a bounding box with an
        input expansion in meters from a given bounding_box

        Subcalls:
         calculate_epsg, convertCoords

        Parameters:
         points (list): output of calc_bbox
         expansion (float): number of meters to expand or shrink the
                            points edges to be

        Returns:
         bl (tuple): x, y of bottom left corner with edges of expansion meters
         tr (tuple): x, y of top right corner with edges of expansion meters
    """
    bl = list(points[0])
    tr = list(points[1])

    epsg = calculate_epsg(bl)

    bl = convertCoords(bl, 4326, epsg)
    tr = convertCoords(tr, 4326, epsg)
    init = [b - a for a,b in zip(bl, tr)]
    distance1 = tr[0] - bl[0]
    distance2 = tr[1] - bl[1]
    expansion1 = (expansion - distance1)/2
    expansion2 = (expansion - distance2)/2
    bl = [bl[0] - expansion1, bl[1] - expansion2]
    tr = [tr[0] + expansion1, tr[1] + expansion2]

    after = [b - a for a,b in zip(bl, tr)]
    print(after)
    if max(init) > 130:
        print("ERROR: Initial field greater than 130m")
    if min(init) < 120:
        print("ERROR: Initial field less than 130m")

    if min(after) < (expansion - 4.5):
        print("ERROR")
    if max(after) > (expansion + 5):
        print("ERROR")
    diffs = [b - a for b, a in zip(after, init)]

    bl = convertCoords(bl, epsg, 4326)
    tr = convertCoords(tr, epsg, 4326)
    return bl, tr

def savi(x, verbose = False):
    # (1.5) * ((08 - 04)/ (08 + 04 + 0.5))
    NIR = x[:, :, :, 6]
    RED = x[:, :, :, 2]
    savis = 1.5 * ( (NIR-RED) / (NIR+RED +0.5))
    if verbose:
        mins = np.min(savis)
        maxs = np.max(savis)
        if mins < -1.0 or maxs > 1.0:
        	print("SAVI: {} {}".format(mins, maxs))
    x = np.concatenate([x, savis[:, :, :, np.newaxis]], axis = -1)
    return x

def msavi2(x, verbose = False):
    # (2 * NIR + 1 - sqrt((2*NIR + 1)^2 - 8*(NIR-RED)) / 2
    NIR = x[:, :, :, 6]
    RED = x[:, :, :, 2]
    for i in range(x.shape[0]):
        NIR_i = x[i, :, :, 6]
        RED_i = x[i, :, :, 2]
        under_sqrt = (2*NIR_i+1)**2 - 8*(NIR_i-RED_i)
        under_sqrt = np.min(under_sqrt)
        if under_sqrt <= 0:
            print("MSAVI2 negative sqrt at: {}, {}".format(i, under_sqrt))
    msavis = (2 * NIR + 1 - np.sqrt( (2*NIR+1)**2 - 8*(NIR-RED) )) / 2
    if verbose:
        mins = np.min(msavis)
        maxs = np.max(msavis)
        if mins < -1 or maxs > 1:
            print("MSAVIS error: {}, {}".format(mins, maxs))
    x = np.concatenate([x, msavis[:, :, :, np.newaxis]], axis = -1)
    return x

def bi(x, verbose = False):
    # (2 + 0 - 1) / (2 + 0 + 1)
    BLUE = x[:, :, :, 0]
    RED = x[:, :, :, 2]
    GREEN = x[:, :, :, 1]
    bis = (BLUE + RED - GREEN) / (BLUE + RED + GREEN)
    if verbose:
        mins = np.min(bis)
        maxs = np.max(bis)
        if mins < -1.5 or maxs > 1.5:
            print("bis error: {}, {}".format(mins, maxs))
    x = np.concatenate([x, bis[:, :, :, np.newaxis]], axis = -1)
    return x

def si(x, verbose = False):
    # (1 - B2) * (1 - B3) * (1 - B4) ** 1/3
    BLUE = x[:, :, :, 0]
    RED = x[:, :, :, 2]
    GREEN = x[:, :, :, 1]
    sis = np.power( (1-BLUE) * (1 - GREEN) * (1 - RED), 1/3)
    if verbose:
        mins = np.min(sis)
        maxs = np.max(sis)
        if mins < -1 or maxs > 1:
            print("sis error: {}, {}".format(mins, maxs))
    x = np.concatenate([x, sis[:, :, :, np.newaxis]], axis = -1)
    return x

def ndmi(x):
    ndmis = [(im[:, :, 5] - im[:, :, 9]) / (im[:, :, 5] + im[:, :, 9]) for im in x]
    ndmis = np.stack(ndmis)
    x = np.concatenate([x, ndmis[:, :, :, np.newaxis]], axis = -1)
    return x

#tf.reset_default_graph()

def Fully_connected(x, units, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=True, units=units)

def Relu(x):
    return tf.nn.relu(x)

def Sigmoid(x):
    return tf.nn.sigmoid(x)

def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')

def Squeeze_excitation_layer(input_x, out_dim, ratio, layer_name):
    with tf.name_scope(layer_name) :
        squeeze = global_avg_pool(input_x)

        excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
        excitation = Relu(excitation)
        excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
        excitation = Sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1,1,1,out_dim])

        scale = input_x * excitation

        return scale

def convGRU(x, cell_fw, cell_bw, ln):
        output, final = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, x, ln, dtype=tf.float32)
        #output = tf.concat(output, -1)
        #final = tf.concat(final, -1)
        return output, final

def remove_blank_steps(array):
    to_update = {}
    sets = []
    for k in range(2):
        for i in range(array.shape[0]):
            for k in range(array.shape[-1]):
                mean = (np.mean(array[i, :, :, k]))
                if mean == 0:
                    print("blank step")
                    sets.append(i)
                    if i < array.shape[0] - 1:
                        array[i, :, :, k] = array[i + 1, :, :, k]
                    else:
                        array[i, :, :, k] = array[i - 1, :, :, k]
                if mean == 1:
                    print("blank step")
                    sets.append(i)
                    if i < array.shape[0] - 1:
                        array[i, :, :, k] = array[i + 1, :, :, k]
                    else:
                        array[i, :, :, k] = array[i - 1, :, :, k]
    for i in range(array.shape[0]):
        for k in range(array.shape[-1]):
            mean = (np.mean(array[i, :, :, k]))
            if mean == 0:
                if i < array.shape[0] - 2:
                    array[i, :, :, k] = array[i + 2, :, :, k]
                else:
                    array[i, :, :, k] = array[i - 2, :, :, k]
            if mean == 1:
                if i < array.shape[0] - 2:
                    array[i, :, :, k] = array[i + 2, :, :, k]
                else:
                    array[i, :, :, k] = array[i - 2, :, :, k]
    return array


def thirty_meter(true, pred, thresh = 0.4):
    subs_pred = pred.reshape(196, 1)
    subs_pred[np.where(subs_pred > thresh)] = 1
    subs_pred[np.where(subs_pred <= thresh)] = 0
    subs_true = true.reshape(196, 1)
    pred = [np.sum(x) for x in subs_pred]
    true = [np.sum(x) for x in subs_true]
    true_positives = []
    false_positives = []
    false_negatives = []
    for p, t in zip(pred, true):
        if p == 1 and t == 1:
            tp = 1
            true_positives.append(tp)
        if p == 1 and t == 0:
            fp = 1
            false_positives.append(fp)
        if p == 0 and t == 1:
            fn = 1
            false_negatives.append(fn)

    if sum(true_positives) + sum(false_positives) > 0:
        prec = sum(true_positives) / (sum(true_positives) + sum(false_positives))
        prec = prec * sum(subs_true)
    else:
        prec = np.nan
    if sum(true_positives) + sum(false_negatives) > 0:
        rec = sum(true_positives) / (sum(true_positives) + sum(false_negatives))
        rec = rec * sum(subs_true)
    else:
        rec = np.nan
    return sum(true_positives), sum(false_positives), sum(false_negatives)#rec, prec, sum(subs_true)

def get_shifts(arr):
    true_m = arr[1:13, 1:13]
    true_l = arr[0:12, 1:13]
    true_r = arr[2:14, 1:13]
    true_u = arr[1:13, 0:12]
    true_d = arr[1:13, 2:14]
    true_dr = arr[2:14, 0:12]
    true_dl = arr[0:12, 0:12]
    true_ur = arr[2:14, 2:14]
    true_ul = arr[0:12, 2:14]
    true_shifts = [true_m, true_l, true_r, true_u, true_d, true_dr, true_dl, true_ur, true_ul]
    return true_shifts



def subset_contiguous_sunny_dates(dates, probs):
    """
    For plots that have at least 24 image dates
    identifies 3-month windows where each month has at least three clean
    image dates, and removes one image per month within the window.
    Used to limit the number of images needed in low-cloud regions from
    a max of 36 to a max of 24.
    """
    begin = [-60, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    end = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 390]
    n_per_month = []
    months_to_adjust = []
    months_to_adjust_again = []
    indices_to_rm = []

    def _indices_month(dates, x, y, indices_to_rm):
        indices_month = np.argwhere(np.logical_and(
                    dates >= x, dates < y)).flatten()
        indices_month = [x for x in indices_month if x not in indices_to_rm]
        return indices_month

    if len(dates) >= 8:
        for x, y in zip(begin, end):
            indices_month = np.argwhere(np.logical_and(
                dates >= x, dates < y)).flatten()
            n_per_month.append(len(indices_month))

        # Convert >3 image months to 2 by removing the cloudiest image
        # Enforces a maximum of 24 images
        for x in range(11):
            # This will only go from 3 images to 2 images
            three_m_sum = np.sum(n_per_month[x:x+3])
            # If at least 4/9 images and a minimum of 0:
            if three_m_sum >= 4 and np.min(n_per_month[x:x+3]) >= 0:
                months_to_adjust += [x, x+1, x+2]

        months_to_adjust = list(set(months_to_adjust))

        # This will sometimes take 3 month windows and cap them to 2 images/month
        for x in [0, 2, 4, 6, 8, 10]:
            three_m_sum = np.sum(n_per_month[x:x+3])
            if three_m_sum >= 5 and np.min(n_per_month[x:x+3]) >= 1:
                if n_per_month[x + 1] == 3: # 3, 3, 3 or 2, 3, 2
                    months_to_adjust_again.append(x + 1)
                elif n_per_month[x] == 3: # 3, 2, 2
                    months_to_adjust_again.append(x)
                elif n_per_month[x + 1] == 2: # 2, 2, 2 or 2, 2, 3
                    months_to_adjust_again.append(x + 1)
                elif n_per_month[x + 2] == 3: # 2, 2, 3
                    months_to_adjust_again.append(x + 2)

        if len(months_to_adjust) > 0:
            for month in months_to_adjust:
                indices_month = np.argwhere(np.logical_and(
                    dates >= begin[month], dates < end[month])).flatten()
                if len(indices_month) > 0:
                    if np.max(probs[indices_month]) >= 0.15:
                        cloudiest_idx = np.argmax(probs[indices_month].flatten())
                    else:
                        cloudiest_idx = 1
                    if len(indices_month) >= 3:
                        indices_to_rm.append(indices_month[cloudiest_idx])

        print(f"Removing {len(indices_to_rm)} sunny dates")
        n_remaining = len(dates) - len(indices_to_rm)

        if len(months_to_adjust_again) > 0 and n_remaining > 12:
            for month in months_to_adjust_again:
                indices_month = np.argwhere(np.logical_and(
                    dates >= begin[month], dates < end[month])).flatten()
                indices_month = [x for x in indices_month if x not in indices_to_rm]
                if np.max(probs[indices_month]) >= 0.15:
                    cloudiest_idx = np.argmax(probs[indices_month].flatten())
                else:
                    cloudiest_idx = 1
                if len(indices_month) >= 2:
                    indices_to_rm.append(indices_month[cloudiest_idx])

        if len(np.argwhere(probs > 0.4)) > 0:
            to_rm = [int(x) for x in np.argwhere(probs > 0.4)]
            print(f"Removing: {dates[to_rm]} missed cloudy dates")
            indices_to_rm.extend(to_rm)

        n_remaining = len(dates) - len(set(indices_to_rm))
        print(f"There are {n_remaining} left, max prob is {np.max(probs)}")

        if n_remaining > 12:
            probs[indices_to_rm] = 0.
            # logic flow here to remove all steps with > 10% cloud cover that leaves 14
            # or if no are >10% cloud cover, to remove the second time step for each month
            # with at least 1 image
            n_above_10_cloud = np.sum(probs >= 0.15)
            print(f"There are {n_above_10_cloud} steps above 10%")
            len_to_rm = (n_remaining - 12)
            print(f"Removing {len_to_rm} steps to leave a max of 12")
            if len_to_rm < n_above_10_cloud:
                max_cloud = np.argpartition(probs, -len_to_rm)[-len_to_rm:]
                print(f"Removing cloudiest dates: {max_cloud}, {probs[max_cloud]}")
                indices_to_rm.extend(max_cloud)
            else:
                # Remove the ones that are > 10% cloud cover
                if n_above_10_cloud > 0:
                    max_cloud = np.argpartition(probs, -n_above_10_cloud)[-n_above_10_cloud:]
                    print(f"Removing cloudiest dates: {max_cloud}, {probs[max_cloud]}")
                    indices_to_rm.extend(max_cloud)

                # And then remove the second time step for months with multiple images
                n_to_remove = len_to_rm - n_above_10_cloud
                print(f"Need to remove {n_to_remove} of {n_remaining}")
                n_removed = 0
                for x, y in zip(begin, end):
                    indices_month = _indices_month(dates, x, y, indices_to_rm)
                    if len(indices_month) > 1:
                        if n_removed <= n_to_remove:
                            if indices_month[1] not in indices_to_rm:
                                indices_to_rm.append(indices_month[1])
                            else:
                                indices_to_rm.append(indices_month[0])
                            print(f"Removing second image in month: {x}, {indices_month[1]}")
                            print(len(set(indices_to_rm)), len(dates))
                        n_removed += 1

        elif np.max(probs) >= 0.20:
            max_cloud = np.argmax(probs)
            print(f"Removing cloudiest date (cloud_removal): {max_cloud}, {probs[max_cloud]}")
            indices_to_rm.append(max_cloud)

        n_remaining = len(dates) - len(set(indices_to_rm))
        print(f"There are {n_remaining} left, max prob is {np.max(probs)}")
        if n_remaining > 10:
            probs[indices_to_rm] = 0.

            # This first block will then remove months 3 and 9 if there are at least 10 months with images
            images_per_month = []
            for x, y, month in zip(begin, end, np.arange(0, 13, 1)):
                indices_month = _indices_month(dates, x, y, set(indices_to_rm))
                images_per_month.append(len(indices_month))
            print(images_per_month)

            months_with_images = np.sum(np.array(images_per_month) >= 1)
            if months_with_images >= 11:
                for x, y, month in zip(begin, end, np.arange(0, 13, 1)):
                    indices_month = _indices_month(dates, x, y, indices_to_rm)
                    if months_with_images >= 12 and len(indices_month) > 0:
                        if (month == 3 or month == 9):
                                indices_to_rm.append(indices_month[0])
                    elif months_with_images == 11 and len(indices_month) > 0:
                        if month == 3:
                            if len(indices_month) > 0:
                                indices_to_rm.append(indices_month[0])

            # This second block will go back through and remove the second image from months with multiple images
            elif np.sum(np.array(images_per_month) >= 2) >= 1:
                n_to_remove = n_remaining - 10
                n_removed = 0
                for x, y in zip(begin, end):
                    indices_month = _indices_month(dates, x, y, indices_to_rm)
                    if len(indices_month) > 1:
                        if n_removed < n_to_remove:
                            if indices_month[1] not in indices_to_rm:
                                indices_to_rm.append(indices_month[1])
                            else:
                                indices_to_rm.append(indices_month[0])
                            print(f"Removing second image in month: {x}, {indices_month[1]}")
                        n_removed += 1

        print(f"Removing {len(indices_to_rm)} sunny/cloudy dates")
    return indices_to_rm
