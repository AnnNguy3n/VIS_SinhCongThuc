import numpy as np
from numba import njit


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
                      INTEREST):
    """
    Output: GeoMax, HarMax, ProMax, ValGeo, GeoLim, ValHar, HarLim, GeoRank, HarRank
    """
    size = INDEX.shape[0] - 1
    arr_inv_idx = np.zeros(size, np.int64)
    for i in range(size):
        start, end = INDEX[i], INDEX[i+1]
        wgt_ = WEIGHT[start:end]
        arr_max = np.where(wgt_==max(wgt_))[0]
        if arr_max.shape[0] == 1:
            arr_inv_idx[i] = start + arr_max[0]
        else:
            arr_inv_idx[i] = -1

    arr_profit = PROFIT[arr_inv_idx]
    arr_profit[arr_inv_idx==-1] = INTEREST
    GeoMax = geomean(arr_profit[1:])
    HarMax = harmean(arr_profit[1:])
    ProMax = arr_profit[0]

    GeoLim = GeoMax
    HarLim = HarMax
    arr_inv_val = WEIGHT[arr_inv_idx]
    arr_inv_val[arr_inv_idx==-1] = 1.7976931348623157e+308
    arr_loop = np.unique(arr_inv_val[1:])
    ValGeo = min(arr_loop)
    delta = max(1e-6*abs(ValGeo), 1e-6)
    ValGeo -= delta
    ValHar = ValGeo

    for v in arr_loop:
        temp_profit = np.where(arr_inv_val > v, arr_profit, INTEREST)
        geo = geomean(temp_profit[1:])
        har = harmean(temp_profit[1:])
        if geo > GeoLim:
            GeoLim = geo
            ValGeo = v

        if har > HarLim:
            HarLim = har
            ValHar = v

    arr_rank = PROFIT_RANK[arr_inv_idx]
    arr_rank = np.where(arr_inv_idx==-1, PROFIT_RANK_NI, arr_rank)
    GeoRank = geomean(arr_rank[1:])
    HarRank = harmean(arr_rank[1:])

    return GeoMax, HarMax, ProMax, ValGeo, GeoLim, ValHar, HarLim, GeoRank, HarRank


@njit
def multi_investment(WEIGHT,
                     INDEX,
                     PROFIT,
                     INTEREST):
    """
    Output: ValGeoNgn, GeoNgn, ValHarNgn, HarNgn
    """
    size = INDEX.shape[0] - 1
    arr_loop = np.zeros((size-1)*5)
    for i in range(1, size):
        start, end = INDEX[i], INDEX[i+1]
        wgt_ = WEIGHT[start:end].copy()
        wgt_[::-1].sort()
        arr_loop[5*(i-1):5*i] = wgt_[:5]

    arr_loop = np.unique(arr_loop)

    ValGeoNgn = -1.0
    GeoNgn = -1.0
    ValHarNgn = -1.0
    HarNgn = -1.0
    temp_profit = np.zeros(size-1)
    for ii in range(len(arr_loop)):
        v = arr_loop[ii]
        bool_wgt = WEIGHT > v
        temp_profit[:] = 0.0
        for i in range(1, size):
            start, end = INDEX[i], INDEX[i+1]
            if np.count_nonzero(bool_wgt[start:end]) == 0:
                temp_profit[i-1] = INTEREST
            else:
                temp_profit[i-1] = PROFIT[start:end][bool_wgt[start:end]].mean()

        geo = geomean(temp_profit)
        har = harmean(temp_profit)
        if geo > GeoNgn:
            GeoNgn = geo
            ValGeoNgn = v

        if har > HarNgn:
            HarNgn = har
            ValHarNgn = v

    return ValGeoNgn, GeoNgn, ValHarNgn, HarNgn


@njit
def multi_investment_strictly(WEIGHT,
                              INDEX,
                              PROFIT,
                              SYMBOL,
                              INTEREST,
                              BOOL_ARG):
    """
    Output: ValGeoNgn2, GeoNgn2, ValHarNgn2, HarNgn2
    """
    size = INDEX.shape[0] - 1
    arr_loop = np.zeros((size-1)*5)
    for i in range(1, size):
        start, end = INDEX[i], INDEX[i+1]
        wgt_ = WEIGHT[start:end].copy()
        wgt_[::-1].sort()
        arr_loop[5*(i-1):5*i] = wgt_[:5]

    arr_loop = np.unique(arr_loop)

    ValGeoNgn2 = -1.0
    GeoNgn2 = -1.0
    ValHarNgn2 = -1.0
    HarNgn2 = -1.0
    temp_profit = np.zeros(size-2)
    for ii in range(len(arr_loop)):
        v = arr_loop[ii]
        bool_wgt = WEIGHT > v
        temp_profit[:] = 0.0
        reason = 0
        for i in range(size-2, 0, -1):
            start, end = INDEX[i], INDEX[i+1]
            inv_cyc_val = bool_wgt[start:end] & BOOL_ARG[start:end]
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

            if len(lst_pro) == 0:
                temp_profit[i-1] = INTEREST
                if np.count_nonzero(inv_cyc_val) == 0:
                    reason = 1
            else:
                temp_profit[i-1] = lst_pro.mean()
                reason = 0

        geo = geomean(temp_profit)
        har = harmean(temp_profit)
        if geo > GeoNgn2:
            GeoNgn2 = geo
            ValGeoNgn2 = v

        if har > HarNgn2:
            HarNgn2 = har
            ValHarNgn2 = v

    return ValGeoNgn2, GeoNgn2, ValHarNgn2, HarNgn2


DESCRIPTION = {
    "single_investment": "GeoMax, HarMax, ProMax, ValGeo, GeoLim, ValHar, HarLim, GeoRank, HarRank".split(", "),
    "multi_investment": "ValGeoNgn, GeoNgn, ValHarNgn, HarNgn".split(", "),
    "multi_investment_strictly": "ValGeoNgn2, GeoNgn2, ValHarNgn2, HarNgn2".split(", ")
}