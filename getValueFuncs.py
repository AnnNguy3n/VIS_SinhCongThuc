import numpy as np
from numba import njit
from evalFuncs import multi_investment_strictly as mis


@njit
def geomean(arr):
    log_sum = sum(np.log(arr))
    return np.exp(log_sum/len(arr))

@njit
def harmean(arr):
    dnmntor = sum(1.0/arr)
    return len(arr)/dnmntor


@njit
def single_investment(WEIGHT,
                      INDEX,
                      PROFIT,
                      PROFIT_RANK,
                      PROFIT_RANK_NI,
                      INTEREST,
                      NUM_CYCLE):
    """
    Output: GeoPro, HarPro, Value, Profit, ValGLim, GeoLim, ValHLim, HarLim, GeoRank, HarRank
    """
    size = INDEX.shape[0] - 1
    arr_profit = np.zeros(size)
    arr_inv_value = np.zeros(size)
    arr_val_rank = np.zeros(size)
    for i in range(size-1, -1, -1):
        idx = size - 1 - i
        start, end = INDEX[i], INDEX[i+1]
        wgt_ = WEIGHT[start:end]
        arr_max = np.where(wgt_==max(wgt_))[0]
        if arr_max.shape[0] == 1:
            arr_profit[idx] = PROFIT[arr_max[0]+start]
            arr_inv_value[idx] = wgt_[arr_max[0]]
            arr_val_rank[idx] = PROFIT_RANK[arr_max[0]+start]
        else:
            arr_profit[idx] = INTEREST
            arr_inv_value[idx] = 1.7976931348623157e+308
            arr_val_rank[idx] = PROFIT_RANK_NI[i]

    GeoRank = np.zeros(NUM_CYCLE)
    HarRank = np.zeros(NUM_CYCLE)
    GeoRank[0] = sum(np.log(arr_val_rank[:-NUM_CYCLE]))
    HarRank[0] = sum(1.0/arr_val_rank[:-NUM_CYCLE])
    for i in range(NUM_CYCLE-1):
        r = arr_val_rank[-NUM_CYCLE+i]
        GeoRank[i+1] = GeoRank[i] + np.log(r)
        HarRank[i+1] = HarRank[i] + 1.0 / r

    Value = arr_inv_value[-NUM_CYCLE:]
    Profit = arr_profit[-NUM_CYCLE:]

    GeoPro = np.zeros(NUM_CYCLE)
    HarPro = np.zeros(NUM_CYCLE)
    GeoPro[0] = sum(np.log(arr_profit[:-NUM_CYCLE]))
    HarPro[0] = sum(1.0/arr_profit[:-NUM_CYCLE])
    for i in range(NUM_CYCLE-1):
        p = arr_profit[-NUM_CYCLE+i]
        GeoPro[i+1] = GeoPro[i] + np.log(p)
        HarPro[i+1] = HarPro[i] + 1.0 / p

    GeoLim = GeoPro.copy()
    HarLim = HarPro.copy()
    ValGLim = np.zeros(NUM_CYCLE)
    ValGLim[0] = min(arr_inv_value[:-NUM_CYCLE])
    for i in range(NUM_CYCLE-1):
        ValGLim[i+1] = min(ValGLim[i], arr_inv_value[-NUM_CYCLE+i])

    ValGLim -= np.maximum(np.abs(ValGLim)*1e-9, 1e-9)
    ValHLim = ValGLim.copy()
    for v in arr_inv_value[:-NUM_CYCLE]:
        temp_profit = np.where(arr_inv_value > v, arr_profit, INTEREST)
        temp_log_sum = sum(np.log(temp_profit[:-NUM_CYCLE-1]))
        temp_dnmntor = sum(1.0/temp_profit[:-NUM_CYCLE-1])
        for i in range(NUM_CYCLE):
            p = temp_profit[-NUM_CYCLE-1+i]
            temp_log_sum += np.log(p)
            temp_dnmntor += 1.0 / p
            if temp_log_sum > GeoLim[i]:
                GeoLim[i] = temp_log_sum
                ValGLim[i] = v

            if temp_dnmntor < HarLim[i]:
                HarLim[i] = temp_dnmntor
                ValHLim[i] = v

    add_id = 0
    for k in range(-NUM_CYCLE, -1):
        add_id += 1
        v = arr_inv_value[k]
        temp_profit = np.where(arr_inv_value > v, arr_profit, INTEREST)
        temp_log_sum = sum(np.log(temp_profit[:k]))
        temp_dnmntor = sum(1.0/temp_profit[:k])
        for i in range(-1-k):
            p = temp_profit[k+i]
            temp_log_sum += np.log(p)
            temp_dnmntor += 1.0 / p
            idx_ = add_id + i
            if temp_log_sum > GeoLim[idx_]:
                GeoLim[idx_] = temp_log_sum
                ValGLim[idx_] = v

            if temp_dnmntor < HarLim[idx_]:
                HarLim[idx_] = temp_dnmntor
                ValHLim[idx_] = v

    results = []
    for i in range(NUM_CYCLE):
        n = size - NUM_CYCLE + i
        result = [
            np.exp(GeoPro[i]/n),
            n / HarPro[i],
            Value[i],
            Profit[i],
            ValGLim[i],
            np.exp(GeoLim[i]/n),
            ValHLim[i],
            n / HarLim[i],
            np.exp(GeoRank[i]/n),
            n / HarRank[i],
        ]
        results.append(result)

    return results


@njit
def multi_investment(WEIGHT,
                     INDEX,
                     PROFIT,
                     INTEREST,
                     NUM_CYCLE,
                     n_val_per_cyc=5):
    """
    Output: Nguong, GeoNgn, HarNgn, ProNgn
    """
    size = INDEX.shape[0] - 1
    arr_loop = np.zeros((size-1)*n_val_per_cyc)
    for i in range(size-1, 0, -1):
        idx = size - 1 - i
        start, end = INDEX[i], INDEX[i+1]
        wgt_ = WEIGHT[start:end].copy()
        wgt_[::-1].sort()
        arr_loop[n_val_per_cyc*idx:n_val_per_cyc*(idx+1)] = wgt_[:n_val_per_cyc]

    if NUM_CYCLE == 1:
        temp_arr_loop = np.unique(arr_loop)
    else:
        temp_arr_loop = np.unique(arr_loop[:n_val_per_cyc*(-NUM_CYCLE+1)])

    Nguong = np.zeros(NUM_CYCLE)
    GeoNgn = np.zeros(NUM_CYCLE)
    HarNgn = np.zeros(NUM_CYCLE)
    temp_profit = np.zeros(size-1)
    for ii in range(len(temp_arr_loop)):
        v = temp_arr_loop[ii]
        bool_wgt = WEIGHT > v
        temp_profit[:] = 0.0
        for i in range(size-1, 0, -1):
            idx = size - 1 - i
            start, end = INDEX[i], INDEX[i+1]
            if np.count_nonzero(bool_wgt[start:end]) == 0:
                temp_profit[idx] = INTEREST
            else:
                temp_profit[idx] = PROFIT[start:end][bool_wgt[start:end]].mean()

        temp_log_sum = sum(np.log(temp_profit[:-NUM_CYCLE]))
        temp_dnmntor = sum(1.0/temp_profit[:-NUM_CYCLE])
        for i in range(NUM_CYCLE):
            p = temp_profit[-NUM_CYCLE+i]
            temp_log_sum += np.log(p)
            temp_dnmntor += 1.0 / p
            if ii == 0:
                Nguong[i] = v
                GeoNgn[i] = temp_log_sum
                HarNgn[i] = temp_dnmntor
            else:
                if temp_log_sum > GeoNgn[i]:
                    Nguong[i] = v
                    GeoNgn[i] = temp_log_sum
                    HarNgn[i] = temp_dnmntor

    add_id = 0
    for k in range(-NUM_CYCLE+1, 0):
        add_id += 1
        if k == -1:
            add_val_loop = np.unique(arr_loop[-n_val_per_cyc:])
        else:
            add_val_loop = np.unique(arr_loop[k*n_val_per_cyc:(k+1)*n_val_per_cyc])

        # add_val_loop = np.setdiff1d(add_val_loop, temp_arr_loop) # Can't njit numba
        add_val_loop = np.array([x for x in add_val_loop if x not in temp_arr_loop])
        temp_arr_loop = np.append(temp_arr_loop, add_val_loop)
        for v in add_val_loop:
            bool_wgt = WEIGHT > v
            temp_profit[:] = 0.0
            for i in range(size-1, 0, -1):
                idx = size - 1 - i
                start, end = INDEX[i], INDEX[i+1]
                if np.count_nonzero(bool_wgt[start:end]) == 0:
                    temp_profit[idx] = INTEREST
                else:
                    temp_profit[idx] = PROFIT[start:end][bool_wgt[start:end]].mean()

            temp_log_sum = sum(np.log(temp_profit[:k]))
            temp_dnmntor = sum(1.0/temp_profit[:k])
            for i in range(-k):
                p = temp_profit[k+i]
                temp_log_sum += np.log(p)
                temp_dnmntor += 1.0 / p
                idx_ = add_id + i
                if temp_log_sum > GeoNgn[idx_]:
                    Nguong[idx_] = v
                    GeoNgn[idx_] = temp_log_sum
                    HarNgn[idx_] = temp_dnmntor

    ProNgn = np.zeros(NUM_CYCLE)
    for i in range(NUM_CYCLE-1, -1, -1):
        idx = NUM_CYCLE - 1 - i
        start, end = INDEX[i], INDEX[i+1]
        v = Nguong[idx]
        mask_ = WEIGHT[start:end] > v
        if np.count_nonzero(mask_) == 0.0:
            ProNgn[idx] = INTEREST
        else:
            ProNgn[idx] = PROFIT[start:end][mask_].mean()

    results = []
    for i in range(NUM_CYCLE):
        n = size - NUM_CYCLE + i
        result = [
            Nguong[i],
            np.exp(GeoNgn[i]/n),
            n / HarNgn[i],
            ProNgn[i]
        ]
        results.append(result)

    return results


@njit
def multi_investment_strictly_1(WEIGHT,
                                INDEX,
                                PROFIT,
                                SYMBOL,
                                INTEREST,
                                BOOL_ARG,
                                NUM_CYCLE=1):
    return [mis(WEIGHT, INDEX, PROFIT, SYMBOL, INTEREST, BOOL_ARG)]


@njit
def multi_investment_strictly(WEIGHT,
                              INDEX,
                              PROFIT,
                              SYMBOL,
                              INTEREST,
                              BOOL_ARG,
                              NUM_CYCLE,
                              n_val_per_cyc=5):
    """
    Output: Nguong2, GeoNgn2, HarNgn2, ProNgn2, hNguong2
    """
    size = INDEX.shape[0] - 1
    arr_loop = np.zeros((size-1)*n_val_per_cyc)
    for i in range(size-1, 0, -1):
        idx = size - 1 - i
        start, end = INDEX[i], INDEX[i+1]
        wgt_ = WEIGHT[start:end].copy()
        wgt_[::-1].sort()
        arr_loop[n_val_per_cyc*idx:n_val_per_cyc*(idx+1)] = wgt_[:n_val_per_cyc]

    if NUM_CYCLE == 1:
        temp_arr_loop = np.unique(arr_loop)
    else:
        temp_arr_loop = np.unique(arr_loop[:n_val_per_cyc*(-NUM_CYCLE+1)])

    Nguong2 = np.zeros(NUM_CYCLE)
    GeoNgn2 = np.zeros(NUM_CYCLE)
    HarNgn2 = np.zeros(NUM_CYCLE)
    ProNgn2 = np.zeros(NUM_CYCLE)
    hNguong2 = np.zeros(NUM_CYCLE)
    temp_profit = np.zeros(size-1)
    bool_arg = BOOL_ARG
    for ii in range(len(temp_arr_loop)):
        v = temp_arr_loop[ii]
        temp_profit[:] = 0.0
        reason = 0
        bool_wgt = WEIGHT > v
        for i in range(size-2, -1, -1):
            start, end = INDEX[i], INDEX[i+1]
            inv_cyc_val = bool_wgt[start:end] & bool_arg[start:end]
            if reason == 0:
                inv_cyc_sym = SYMBOL[start:end]
                end2 = INDEX[i+2]
                pre_cyc_val = bool_wgt[end:end2]
                pre_cyc_sym = SYMBOL[end:end2]
                coms = np.intersect1d(pre_cyc_sym[pre_cyc_val], inv_cyc_sym[inv_cyc_val])
                isin = np.full(end-start, False)
                for j in range(end-start):
                    if inv_cyc_sym[j] in coms:
                        isin[j] = True
                lst_pro = PROFIT[start:end][isin]
            else:
                lst_pro = PROFIT[start:end][inv_cyc_val]

            _idx_ = size - 2 - i
            if len(lst_pro) == 0:
                temp_profit[_idx_] = INTEREST
                if np.count_nonzero(inv_cyc_val) == 0:
                    reason = 1
            else:
                temp_profit[_idx_] = np.mean(lst_pro)
                reason = 0

        temp_log_sum = sum(np.log(temp_profit[:-NUM_CYCLE-1]))
        temp_dnmntor = sum(1.0/temp_profit[:-NUM_CYCLE-1])
        for i in range(NUM_CYCLE):
            p = temp_profit[-NUM_CYCLE-1+i]
            temp_log_sum += np.log(p)
            temp_dnmntor += 1.0 / p
            if ii == 0:
                Nguong2[i] = v
                GeoNgn2[i] = temp_log_sum
                HarNgn2[i] = temp_dnmntor
                ProNgn2[i] = temp_profit[-NUM_CYCLE+i]
                hNguong2[i] = v
            else:
                if temp_log_sum > GeoNgn2[i]:
                    Nguong2[i] = v
                    GeoNgn2[i] = temp_log_sum
                    # HarNgn2[i] = temp_dnmntor
                    ProNgn2[i] = temp_profit[-NUM_CYCLE+i]

                if temp_dnmntor < HarNgn2[i]:
                    HarNgn2[i] = temp_dnmntor
                    hNguong2[i] = v

    add_id = 0
    for k in range(-NUM_CYCLE+1, 0):
        add_id += 1
        if k == -1:
            add_val_loop = np.unique(arr_loop[-n_val_per_cyc:])
        else:
            add_val_loop = np.unique(arr_loop[k*n_val_per_cyc:(k+1)*n_val_per_cyc])

        # add_val_loop = np.setdiff1d(add_val_loop, temp_arr_loop) # Can't njit numba
        add_val_loop = np.array([x for x in add_val_loop if x not in temp_arr_loop])
        temp_arr_loop = np.append(temp_arr_loop, add_val_loop)
        for v in add_val_loop:
            temp_profit[:] = 0.0
            reason = 0
            bool_wgt = WEIGHT > v
            for i in range(size-2, -1, -1):
                start, end = INDEX[i], INDEX[i+1]
                inv_cyc_val = bool_wgt[start:end] & bool_arg[start:end]
                if reason == 0:
                    inv_cyc_sym = SYMBOL[start:end]
                    end2 = INDEX[i+2]
                    pre_cyc_val = bool_wgt[end:end2]
                    pre_cyc_sym = SYMBOL[end:end2]
                    coms = np.intersect1d(pre_cyc_sym[pre_cyc_val], inv_cyc_sym[inv_cyc_val])
                    isin = np.full(end-start, False)
                    for j in range(end-start):
                        if inv_cyc_sym[j] in coms:
                            isin[j] = True
                    lst_pro = PROFIT[start:end][isin]
                else:
                    lst_pro = PROFIT[start:end][inv_cyc_val]

                _idx_ = size - 2 - i
                if len(lst_pro) == 0:
                    temp_profit[_idx_] = INTEREST
                    if np.count_nonzero(inv_cyc_val) == 0:
                        reason = 1
                else:
                    temp_profit[_idx_] = np.mean(lst_pro)
                    reason = 0

            temp_log_sum = sum(np.log(temp_profit[:k-1]))
            temp_dnmntor = sum(1.0/temp_profit[:k-1])
            for i in range(-k):
                p = temp_profit[k-1+i]
                temp_log_sum += np.log(p)
                temp_dnmntor += 1.0 / p
                idx_ = add_id + i
                if temp_log_sum > GeoNgn2[idx_]:
                    Nguong2[idx_] = v
                    GeoNgn2[idx_] = temp_log_sum
                    # HarNgn2[idx_] = temp_dnmntor
                    ProNgn2[idx_] = temp_profit[k+i]

                if temp_dnmntor < HarNgn2[idx_]:
                    HarNgn2[idx_] = temp_dnmntor
                    hNguong2[idx_] = v

    results = []
    for i in range(NUM_CYCLE):
        n = size - NUM_CYCLE + i - 1
        result = [
            Nguong2[i],
            np.exp(GeoNgn2[i]/n),
            n / HarNgn2[i],
            ProNgn2[i],
            hNguong2[i]
        ]
        results.append(result)

    return results


@njit
def find_slope(WEIGHT, INDEX, PROFIT, INTEREST, NUM_CYCLE):
    """
    Output: slope_avg, slope_wgtavg
    """
    size = INDEX.shape[0] - 1
    arr_profit = np.zeros(size)
    arr_inv_value = np.zeros(size)
    for i in range(size-1, -1, -1):
        idx = size - 1 - i
        start, end = INDEX[i], INDEX[i+1]
        wgt_ = WEIGHT[start:end]
        arr_max = np.where(wgt_==max(wgt_))[0]
        if arr_max.shape[0] == 1:
            arr_profit[idx] = PROFIT[arr_max[0]+start]
            arr_inv_value[idx] = wgt_[arr_max[0]]
        else:
            arr_profit[idx] = INTEREST
            arr_inv_value[idx] = 1.7976931348623157e+308

    results = []
    for i in range(-NUM_CYCLE, 0):
        temp_profit = arr_profit[:i]
        temp_value = arr_inv_value[:i]
        y = arr_inv_value[i]
        slope_avg, slope_wgtavg = _find_slope(temp_profit, temp_value, y)
        results.append([slope_avg, slope_wgtavg])

    return results

@njit
def linear_regression(A, B):
    try:
        # Calculate means
        mean_A = np.mean(A)
        mean_B = np.mean(B)

        # Calculate covariance and variance
        cov_AB = 0.0
        for i in range(len(A)):
            cov_AB += (A[i]-mean_A)*(B[i]-mean_B)

        cov_AB /= len(A)
        var_A = np.var(A)

        # Estimate coefficients
        m = cov_AB / var_A
        b = mean_B - m * mean_A

        return m, b
    except:
        return 0.0, 0.0

@njit
def _find_slope(profit_, value_, y):
    if (value_ == 1.7976931348623157e+308).any() or y == 1.7976931348623157e+308:
        return 0.0, 0.0

    temp = np.argsort(value_)
    value = value_[temp]
    profit = profit_[temp]
    n = value.shape[0]
    arr_avg = np.zeros(n)
    for i in range(n):
        arr_avg[i] = np.mean(profit[i:])

    m1, b1 = linear_regression(value, arr_avg)
    slope_avg = m1*y + b1
    if np.isinf(slope_avg):
        slope_avg = 0.0

    if (value_ <= 0.0).any() or y <= 0.0:
        return slope_avg, 0.0

    arr_wgtavg = np.zeros(n)
    for i in range(n):
        arr_wgtavg[i] = np.sum(profit[i:] * value[i:]) / np.sum(value[i:])

    m2, b2 = linear_regression(value, arr_wgtavg)
    slope_wgtavg = m2*y + b2
    if np.isinf(slope_wgtavg):
        slope_wgtavg = 0.0

    return slope_avg, slope_wgtavg


@njit
def sinhF_multi_investment_strictly(WEIGHT,
                                    INDEX,
                                    PROFIT,
                                    SYMBOL,
                                    INTEREST,
                                    BOOL_ARG):
    """
    Output: GeoNgn2
    """
    return multi_investment_strictly(WEIGHT, INDEX, PROFIT, SYMBOL, INTEREST, BOOL_ARG, 1, 5)[0][1]


DESCRIPTION = {
    "single_investment": "GeoPro, HarPro, Value, Profit, ValGLim, GeoLim, ValHLim, HarLim, GeoRank, HarRank".split(", "),
    "multi_investment": "Nguong, GeoNgn, HarNgn, ProNgn".split(", "),
    "multi_investment_strictly": "Nguong2, GeoNgn2, HarNgn2, ProNgn2, hNguong2".split(", "),
    "find_slope": "Slope_avg, Slope_wgtavg".split(", "),
    "multi_investment_strictly_1": "ValGeoNgn2, GeoNgn2, ValHarNgn2, HarNgn2".split(", ")
    # "sinhF_multi_investment_strictly": ["GeoNgn2"]
}
