from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from sklearn.cross_decomposition import CCA
%run ../src/pca/pca_filter.py

def load_reference_years(input_year, target_year):
    inp = np.empty((24, 5, 5, 142, 142, 17))
    targ = np.empty((24, 5, 5, 142, 142, 17))
    for x in tnrange(0, 5):
        for y in range(0, 5):
            inp_x = hkl.load(f"../tile_data/{LANDSCAPE}/{str(input_year)}/processed/{str(x)}/{str(y)}.hkl")
            targ_x = hkl.load(f"../tile_data/{LANDSCAPE}/{str(target_year)}/processed/{str(x)}/{str(y)}.hkl")
            inp[:, x, y, ...] = inp_x[...,]
            targ[:, x, y, ...] = targ_x[...]
    inp = inp.reshape(24, 5*5*142*142, 17)
    targ = targ.reshape(24, 5*5*142*142, 17)
    return inp, targ

def identify_pif_pca(input_year, reference):
    pif_mask = numpy.ones((142* 142*5*5), dtype=numpy.bool)
    for date in range(0, 24, 2):
        for band in range(0, 5):
            pif_band_mask = pca_fit_and_filter_pixel_list(input_year[date, ..., band].flatten(),
                                            reference[date, ..., band].flatten(),
                                            660)
            pif_mask = numpy.logical_and(pif_mask, pif_band_mask)
            print(np.sum(pif_mask))
    return np.argwhere(pif_mask == True)

def identify_pifs_new(input_year, reference):
    x = input_year[:, :, :10]
    x = np.swapaxes(x, 0, 1).reshape(142*142*5*5, 24*10)
    y = reference[:, ..., :10]
    y = np.swapaxes(y, 0, 1).reshape(142*142*5*5, 24*10)
    cca = CCA(n_components=2)
    xs, ys = cca.fit_transform(x, y)
    diffs = abs(xs - ys)
    diffs = np.sum(diffs**2, axis = 1)
    diffs = (diffs - np.mean(diffs)) / np.std(diffs)
    diffs = np.argwhere(diffs > np.percentile(diffs, 99.9))
    return diffs#np.argwhere(diffs < np.percentile(diffs, 1))

def identify_pifs_pairs(input_year, reference):
    diffs_all = np.zeros(142*142*5*5)
    for date in tnrange(0, 24):
        x = input_year[date, ..., :10].reshape(142*142*5*5, 10)
        y = reference[date, ..., :10].reshape(142*142*5*5, 10)
        cca = CCA(n_components=2)
        cca.fit(x, y)
        xs = cca.transform(x)
        ys = cca.transform(y)
        diffs = np.sum(abs(xs - ys)**2, axis = 1)
        diffs = diffs.reshape(142*142*5*5)
        diffs_all += diffs
    print(diffs_all.shape)
    diffs = np.argwhere(diffs_all < np.percentile(diffs_all, 0.1))
    return diffs

def identify_pifs(input_year, reference):
    pearsons = np.empty(142*142*5*5)
    input_year = input_year[..., 3].reshape(24, 142*142*5*5)
    reference = reference[..., 3].reshape(24, 142*142*5*5)
    
    for i in tnrange(pearsons.shape[0]):
        pearsons[i] = pearsonr(input_year[:, i], reference[:, i])[0]
    
    pifs = np.argwhere(pearsons > np.percentile(pearsons, 99))
    #stacked = np.concatenate([input_year, reference], axis = 0)
    #stacked = stacked[..., 3].reshape(48, 142*142*5*5)
    #stacked = np.percentile(stacked, 75, 0) - np.percentile(stacked, 25, 0)
    #pifs = np.argwhere(stacked < np.percentile(stacked, 2))
    #diffs = abs( 1 - (abs(input_year[..., 3] / reference[..., 3])) )#
    #diffs = diffs.reshape(24, 142*142)
    #diffs = np.std(diffs, axis = (0))
    #pifs = np.argwhere((diffs < 0.05))
    return pifs, pearsons


def linear_adjust_pif(pifs, small_input, small_target, large_input, large_target):
    output = small_input.copy()
    
    for date in range(0, 24):
        for band in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]:
            input_ = large_input[:, pifs, band]
            target = large_target[:, pifs, band]
            
            target_date = target[date].squeeze()[:, np.newaxis]
            input_date = input_[date].squeeze()[:, np.newaxis]
            reg = LinearRegression().fit(input_date, target_date)
            input_updated = reg.predict(small_input[date, ..., band].reshape(142*142, 1))

            output[date, ..., band] = input_updated.reshape(1, 142, 142)
    return output

#sns.scatterplot([x for x in range(len(diffs))], sorted(diffs))