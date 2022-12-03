import pandas as pd
import numpy as np
import os
from numba import njit


@njit
def _get_profit_by_weight(weight, profit, index):
    temp_profit = 1.0
    for i in range(index.shape[0]-2, -1, -1):
        temp = weight[index[i]:index[i+1]]
        max_ = np.where(temp == np.max(temp))[0] + index[i]
        if max_.shape[0] == 1:
            temp_profit *= profit[max_[0]]

    return temp_profit**(1.0/(index.shape[0]-1))


@njit
def _calculate_formula(formula, operand):
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
    def __init__(self, data:pd.DataFrame, pathSaveFormula:str, soChuKyTrain:int) -> None:
        '''
        Lưu ý về data đầu vào:
        --------------------------------------------------
            * Phải có 3 cột (tên in hoa) là "TIME", "PROFIT" và "SYMBOL".
            * Các bản ghi phải được sắp xếp theo thứ tự giảm dần của cột "TIME" trước khi truyền vào.
            * Ngoài các cột "TIME", "PROFIT", "SYMBOL" và "EXCHANGE" (nếu có), các cột khác được coi là các biến để thử công thức.
        '''

        start_time = np.min(data["TIME"])
        last_time = start_time + soChuKyTrain - 1

        self.__TRAIN_DATA = data[(data["TIME"] >= start_time) & (data["TIME"] <= last_time)]
        self.__TEST_DATA = data[data["TIME"] == last_time + 1]

        self.__PROFIT = np.array(self.__TRAIN_DATA["PROFIT"], dtype=np.float64)
        self.__TEST_PROFIT = np.array(self.__TEST_DATA["PROFIT"], dtype=np.float64)

        drop_columns = ["TIME", "PROFIT", "SYMBOL"]
        try:
            self.__TRAIN_DATA["EXCHANGE"]
            drop_columns.append("EXCHANGE")
        except:
            pass

        self.__OPERAND = np.transpose(np.array(self.__TRAIN_DATA.drop(columns=drop_columns), dtype=np.float64))
        self.__TEST_OPERAND = np.transpose(np.array(self.__TEST_DATA.drop(columns=drop_columns), dtype=np.float64))

        if not os.path.exists(pathSaveFormula):
            raise Exception(f"Không tồn tại đường dẫn tới thư mục để lưu công thức, {pathSaveFormula}")
        else:
            self.__path = pathSaveFormula
            if not self.__path.endswith('/'):
                self.__path += '/'

        time_arr = np.array(self.__TRAIN_DATA["TIME"])
        qArr = np.unique(time_arr)
        self.__INDEX = np.full(qArr.shape[0]+1, 0, dtype=np.int64)
        for i in range(qArr.shape[0]):
            if i == qArr.shape[0] - 1:
                self.__INDEX[qArr.shape[0]] = time_arr.shape[0]
            else:
                temp = time_arr[self.__INDEX[i]]
                for j in range(self.__INDEX[i], time_arr.shape[0]):
                    if time_arr[j] != temp:
                        self.__INDEX[i+1] = j
                        break
    @property
    def TRAIN_DATA(self):
        return self.__TRAIN_DATA.copy()
    @property
    def TEST_DATA(self):
        return self.__TEST_DATA.copy()
    @property
    def path(self):
        return self.__path
    @path.setter
    def path(self, path):
        if not os.path.exists(path):
            raise Exception(f"Không tồn tại đường dẫn tới thư mục để lưu công thức, {path}")
        else:
            self.__path = path


    def convert_formula_to_str(self, formula):
        temp = "+-*/"
        str_formula = ""
        for i in range(formula.shape[0]):
            if i % 2 == 1:
                str_formula += str(formula[i])
            else:
                str_formula += temp[formula[i]]

        return str_formula


    def convert_str_to_formula(self, str_formula):
        temp = "+-*/"
        f_len = sum(str_formula.count(c) for c in temp) * 2
        str_len = len(str_formula)
        if self.__OPERAND.shape[0] <= 256:
            formula = np.full(f_len, 0, dtype=np.uint8)
        else:
            formula = np.full(f_len, 0, dtype=np.uint16)

        idx = 0
        for i in range(f_len):
            if i % 2 == 1:
                t_ = 0
                while True:
                    t_ = 10*t_ + int(str_formula[idx])
                    idx += 1
                    if idx == str_len or str_formula[idx] in temp:
                        break

                formula[i] = t_
            else:
                formula[i] = temp.index(str_formula[idx])
                idx += 1

        return formula


    def get_formula_geomean_profit(self, formula):
        if type(formula) == str:
            formula = self.convert_str_to_formula(formula)

        return _get_profit_by_weight(_calculate_formula(formula, self.__OPERAND), self.__PROFIT, self.__INDEX)
    

    def convert_npy_file_to_DataFrame(self, path):
        list_formula = np.load(path, allow_pickle=True)
        list_str_formula = []
        list_profit = []
        list_next_invest = []
        list_next_profit = []

        temp_symbol = self.__TEST_DATA["SYMBOL"]
        temp_profit = self.__TEST_DATA["PROFIT"]

        for i in range(list_formula.shape[0]):
            formula = list_formula[i]
            list_str_formula.append(self.convert_formula_to_str(formula))
            list_profit.append(self.get_formula_geomean_profit(formula))
            weight = _calculate_formula(formula, self.__TEST_OPERAND)
            max_ = np.where(weight == np.max(weight))[0]
            if max_.shape[0] == 1:
                next_invest = temp_symbol.iloc[max_[0]]
                next_profit = temp_profit.iloc[max_[0]]
            else:
                next_invest = "NOT_INVEST_2"
                next_profit = 1.0
            
            list_next_invest.append(next_invest)
            list_next_profit.append(next_profit)
        
        return pd.DataFrame({
            "formula": list_str_formula,
            "geomean profit": list_profit,
            "invest": list_next_invest,
            "profit": list_next_profit
        })


    def explain_formula(self, formula):
        weight = _calculate_formula(formula, self.__OPERAND)
        temp_profit = 1.0
        for i in range(self.__INDEX.shape[0]-2, -1, -1):
            temp = weight[self.__INDEX[i]:self.__INDEX[i+1]]
            max_ = np.where(temp == np.max(temp))[0] + self.__INDEX[i]
            if max_.shape[0] == 1:
                temp_profit *= self.__PROFIT[max_[0]]
                print("Quý thứ", self.__INDEX.shape[0]-1-i, "đầu tư", self.__TRAIN_DATA.iloc[max_[0]]["SYMBOL"], "lãi", self.__PROFIT[max_[0]])
            else:
                print("Quý thứ", self.__INDEX.shape[0]-1-i, "không đầu tư")
        
        weight = _calculate_formula(formula, self.__TEST_OPERAND)
        max_ = np.where(weight == np.max(weight))[0]
        if max_.shape[0] == 1:
            temp_profit_2 = temp_profit * self.__TEST_PROFIT[max_[0]]
            print("Quý thứ", self.__INDEX.shape[0], "đầu tư", self.__TEST_DATA.iloc[max_[0]]["SYMBOL"], "lãi", self.__TEST_PROFIT[max_[0]])
        else:
            temp_profit_2 = temp_profit
            print("Quý thứ", self.__INDEX.shape[0], "không đầu tư")
        
        print("Lợi nhuận trung bình nhân (chưa tính lần đầu tư cuối):", temp_profit**(1.0/(self.__INDEX.shape[0]-1)))
        print("Lợi nhuận trung bình nhân (đã tính lần đầu tư cuối):", temp_profit_2**(1.0/(self.__INDEX.shape[0])))