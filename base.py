import pandas as pd
import numpy as np
from numba import njit
import os


@njit
def _get_profit_by_weight(weight, profit, index):
    temp_profit = 1.0
    for i in range(index.shape[0]-2, 0, -1):
        temp = weight[index[i]:index[i+1]]
        max_ = np.where(temp == np.max(temp))[0] + index[i]
        if max_.shape[0] == 1:
            temp_profit *= profit[max_[0]]

    return temp_profit**(1.0/(index.shape[0]-2))


@njit
def _calculate_formula(formula, operand):
    '''
    * formula: Dạng numpy array, độ dài chẵn, là biểu diễn dạng số của n toán tử và n toán hạng xen kẽ nhau.
        * Toán hạng: 0,1,2,...,N-1 với N là số toán hạng có thể cho vào công thức.
        * Toán tử: 0,1,2,3 lần lượt là cộng, trừ, nhân, chia.
    '''
    temp_0 = np.zeros(operand.shape[1])
    temp_1 = temp_0.copy()
    temp_op = -1

    for i in range(1, formula.shape[0], 2):
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


class Method:
    def __init__(self, data: pd.DataFrame, pathSaveFormula: str) -> None:
        self.__TIME = np.array(data["TIME"], dtype=np.int64)
        self.__PROFIT = np.array(data["PROFIT"], dtype=np.float64)
        self.__SYMBOL = data["SYMBOL"]
        self.__EXCHANGE = data["EXCHANGE"]
        self.__OPERAND = np.transpose(np.array(data.drop(columns=["TIME", "PROFIT", "SYMBOL", "EXCHANGE"]), dtype=np.float64))
        if not os.path.exists(pathSaveFormula):
            raise Exception("Đường dẫn đến thư mục để lưu công thức không tồn tại.")

        self.path = pathSaveFormula

        qArr = np.unique(self.__TIME)
        self.__INDEX = np.full(qArr.shape[0]+1, 0, dtype=np.int64)
        for i in range(qArr.shape[0]):
            if i == qArr.shape[0] - 1:
                self.__INDEX[qArr.shape[0]] = self.__TIME.shape[0]
            else:
                temp = self.__TIME[self.__INDEX[i]]
                for j in range(self.__INDEX[i], self.__TIME.shape[0]):
                    if self.__TIME[j] != temp:
                        self.__INDEX[i+1] = j
                        break
    @property
    def TIME(self):
        return self.__TIME.copy()
    @property
    def PROFIT(self):
        return self.__PROFIT.copy()
    @property
    def SYMBOL(self):
        return self.__SYMBOL.copy()
    @property
    def EXCHANGE(self):
        return self.__EXCHANGE.copy()
    @property
    def OPERAND(self):
        return self.__OPERAND.copy()
    @property
    def INDEX(self):
        return self.__INDEX.copy()


    def convert_str_to_formula(self, str_formula, var_char="ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        '''
        Chuyển công thức từ dạng string về dạng array.
        '''
        temp = "+-*/"
        strlen = len(str_formula)
        if self.__OPERAND.shape[0] <= 256:
            formula = np.full(strlen, 0, dtype=np.uint8)
        else:
            formula = np.full(strlen, 0, dtype=np.uint16)

        for i in range(strlen):
            if i % 2 == 1:
                formula[i] = var_char.index(str_formula[i])
            else:
                formula[i] = temp.index(str_formula[i])

        return formula


    def convert_formula_to_str(self, formula, var_char="ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        '''
        Chuyển công thức từ dạng array về dạng string.
        '''
        temp = "+-*/"
        str_formula = ""

        for i in range(formula.shape[0]):
            if i % 2 == 1:
                str_formula += var_char[formula[i]]
            else:
                str_formula += temp[formula[i]]

        return str_formula


    def get_formula_profit(self, formula, var_char="ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        '''
        * Chấp nhận công thức cả ở dạng array lẫn dạng string.
        * formula: Dạng numpy array, độ dài chẵn, là biểu diễn dạng số của n toán tử và n toán hạng xen kẽ nhau.
            * Toán hạng: 0,1,2,...,N-1 với N là số toán hạng có thể cho vào công thức.
            * Toán tử: 0,1,2,3 lần lượt là cộng, trừ, nhân, chia.
        '''
        if type(formula) == str:
            formula = self.convert_str_to_formula(formula, var_char)

        return _get_profit_by_weight(_calculate_formula(formula, self.__OPERAND), self.__PROFIT, self.__INDEX)


    def convert_npy_file_to_DataFrame(self, path):
        file = np.load(path, allow_pickle=True)
        list_formula = file[1]
        list_formula_profit = file[2]

        temp = []
        for i in range(len(file[1])):
            temp.append(self.convert_formula_to_str(list_formula[i]))

        return pd.DataFrame({
            "formula": temp,
            "array": list(file[1]),
            "profit": list(file[2])
        })
