import numpy as np
import pandas as pd
from numba import njit


class Base:
    def __init__(self, data: pd.DataFrame, interest: float) -> None:
        data = data.reset_index(drop=True)
        data.fillna(0.0, inplace=True)

        # Check cac cot bat buoc
        drop_cols = ["TIME", "PROFIT", "SYMBOL", "VALUEARG"]
        for col in drop_cols:
            if col not in data.columns:
                raise Exception(f"Thieu cot {col}")

        # Check dtype cua TIME, PROFIT va VALUEARG
        if data["TIME"].dtype != "int64":
            dtype_ = data["TIME"].dtype
            raise Exception(f"dtype cua TIME khong phai int64 ({dtype_})")

        if data["PROFIT"].dtype != "float64":
            dtype_ = data["PROFIT"].dtype
            raise Exception(f"dtype cua PROFIT khong phai float64 ({dtype_})")

        if data["VALUEARG"].dtype != "float64" and data["VALUEARG"].dtype != "int64":
            dtype_ = data["VALUEARG"].dtype
            raise Exception(f"dtype cua VALUEARG khong phai float64 ({dtype_})")

        # Check thu tu cot TIME
        if data["TIME"].diff().max() > 0:
            raise Exception("Cot TIME phai giam dan")

        # INDEX
        time_uni = data["TIME"].unique()
        index = []
        for i in range(data["TIME"].max(), data["TIME"].min()-1, -1):
            if i not in time_uni:
                raise Exception(f"Thieu chu ky {i}")

            index.append(data[data["TIME"]==i].index[0])

        index.append(data.shape[0])
        self.INDEX = np.array(index)

        # Loai cac cot co kieu du lieu khong phai int64 va float64
        for col in data.columns:
            if col not in drop_cols and data[col].dtype not in ["int64", "float64"]:
                drop_cols.append(col)

        self.drop_cols = drop_cols

        # Attrs
        self.data = data
        self.INTEREST = interest
        self.PROFIT = np.array(data["PROFIT"], float)
        self.PROFIT[self.PROFIT < 5e-324] = 5e-324
        self.VALUEARG = np.array(data["VALUEARG"], float)

        symbol_name = data["SYMBOL"].unique()
        self.symbol_name = {symbol_name[i]:i for i in range(len(symbol_name))}
        self.SYMBOL = np.array([self.symbol_name[s] for s in data["SYMBOL"]])
        self.symbol_name = {v:k for k,v in self.symbol_name.items()}

        operand_data = data.drop(columns=drop_cols)
        operand_name = operand_data.columns
        self.operand_name = {i:operand_name[i] for i in range(len(operand_name))}
        self.OPERAND = np.transpose(np.array(operand_data, float))

        self.PROFIT_RANK = np.zeros(data.shape[0])
        self.PROFIT_RANK_NI = np.zeros(self.INDEX.shape[0]-1)
        temp_serie = pd.Series([self.INTEREST])
        for i in range(self.INDEX.shape[0]-1):
            start, end = self.INDEX[i], self.INDEX[i+1]
            temp_ = pd.concat([data.loc[start:end-1, "PROFIT"], temp_serie], ignore_index=True)
            temp_rank = np.array(temp_.rank(method="min"), float) / (end-start+1)
            self.PROFIT_RANK[start:end] = temp_rank[:-1]
            self.PROFIT_RANK_NI[i] = temp_rank[-1]


__STRING_OPERATOR = "+-*/"

def convert_arrF_to_strF(arrF):
    strF = ""
    for i in range(len(arrF)):
        if i % 2 == 1:
            strF += str(arrF[i])
        else:
            strF += __STRING_OPERATOR[arrF[i]]

    return strF

def convert_strF_to_arrF(strF):
    f_len = sum(strF.count(c) for c in __STRING_OPERATOR) * 2
    str_len = len(strF)
    arrF = np.full(f_len, 0)

    idx = 0
    for i in range(f_len):
        if i % 2 == 1:
            t_ = 0
            while True:
                t_ = 10*t_ + int(strF[idx])
                idx += 1
                if idx == str_len or strF[idx] in __STRING_OPERATOR:
                    break

            arrF[i] = t_
        else:
            arrF[i] = __STRING_OPERATOR.index(strF[idx])
            idx += 1

    return arrF


def similarity_filter(df_CT, fml_col, n=100, level=2):
    list_CT = []
    for ct in df_CT[fml_col]:
        if type(ct) == str:
            list_CT.append(convert_strF_to_arrF(ct))
        else:
            list_CT.append(ct)

    list_index = _similarity_filter(list_CT, n, level)
    return df_CT.iloc[list_index].reset_index(drop=True)


@njit
def check_similar_2(f1_, f2_, level):
    f1 = np.unique(f1_[1::2])
    f2 = np.unique(f2_[1::2])

    if len(f1) > len(f2):
        F1 = f1
        F2 = f2
    else:
        F1 = f2
        F2 = f1

    count = 0
    for i in F1:
        if i not in F2:
            count += 1

    if count >= level:
        return False

    return True


@njit
def _similarity_filter(list_ct, num_CT, level):
    list_index = [0]
    count = 1
    for i in range(1, len(list_ct)):
        check = True
        for j in list_index:
            if check_similar_2(list_ct[i], list_ct[j], level):
                check = False
                break

        if check:
            list_index.append(i)
            count += 1
            if count == num_CT:
                print(i)
                break

    return list_index


@njit
def calculate_formula(formula, operand):
    temp_0 = np.zeros(operand.shape[1])
    temp_1 = temp_0.copy()
    temp_op = -1
    num_operand = operand.shape[0]
    for i in range(1, formula.shape[0], 2):
        if formula[i] >= num_operand:
            raise

        if formula[i-1] < 2:
            temp_op = formula[i-1]
            temp_1 = operand[formula[i]].copy()
        else:
            if formula[i-1] == 2:
                temp_1 *= operand[formula[i]]
            else:
                temp_1 /= operand[formula[i]]

        if i+1 == formula.shape[0] or formula[i+1] < 2:
            if temp_op == 0:
                temp_0 += temp_1
            else:
                temp_0 -= temp_1

    temp_0[np.isnan(temp_0)] = -1.7976931348623157e+308
    temp_0[np.isinf(temp_0)] = -1.7976931348623157e+308
    return temp_0


@njit
def decode_formula(f, len_):
    rs = np.full(len(f)*2, 0, dtype=int)
    rs[0::2] = f // len_
    rs[1::2] = f % len_
    return rs


@njit
def encode_formula(f, len_):
    return f[0::2]*len_ + f[1::2]


@njit
def get_opposite_formula(fs_):
    formulas = fs_.copy()
    M, N = formulas.shape
    for i in range(M):
        for k in range(0, N, 2):
            if formulas[i, k] == 0:
                formulas[i, k] = 1
            elif formulas[i, k] == 1:
                formulas[i, k] = 0

    return formulas
