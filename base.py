import pandas as pd
import numpy as np
import os
from colorama import Fore, Style
import warnings
warnings.filterwarnings("ignore")

from nopy import _get_profit_by_weight, _calculate_formula

class Method:
    def __init__(self, data:pd.DataFrame, path_save_formula:str, so_chu_ky_train:int) -> None:
        # Check các cột bắt buộc
        drop_cols = ["TIME", "PROFIT", "SYMBOL"]
        for col in drop_cols:
            if col not in data.columns:
                raise Exception(f'Thiếu cột "{col}".')

        # Check kiểu dữ liệu của cột TIME và PROFIT
        if data["TIME"].dtypes != "int64":
            raise Exception(f'Kiểu dữ liệu của cột "TIME" phải là int64 (hiện tại đang là {data["TIME"].dtypes}).')
        if data["PROFIT"].dtypes != "float64":
            raise Exception(f'Kiểu dữ liệu của cột "PROFIT" phải là float64 (hiện tại đang là {data["PROFIT"].dtypes}).')

        # Check cột TIME xem có tăng dần không
        if data["TIME"].diff().max() > 0:
            raise Exception(f'Dữ liệu phải được sắp xếp theo sự giảm dần của cột "TIME".')

        # Check các cột cần được drop
        for col in data.columns:
            if col not in drop_cols and data[col].dtypes == "object":
                drop_cols.append(col)

        print(Fore.LIGHTYELLOW_EX + f"Cảnh báo: Các cột không được coi là biến để sinh công thức: {drop_cols}. Nếu danh sách trên có một cột cần được coi là biến, hãy kiểm tra lại kiểu dữ liệu của cột.\n" , Style.RESET_ALL)

        # Kiểm tra xem path có tồn tại hay không
        if type(path_save_formula) != str or not os.path.exists(path_save_formula):
            raise Exception(f'Không tồn tại thư mục {path_save_formula}/.')
        else:
            if not path_save_formula.endswith("/") and not path_save_formula.endswith("\\"):
                path_save_formula += "/"
                print(Fore.LIGHTBLUE_EX + f'Một dấu "/" đã được tự động thêm vào đường dẫn tới thư mục lưu công thức. Đường dẫn mới là: {path_save_formula}.\n', Style.RESET_ALL)

        # Thiết lập các thuộc tính
        start_time = np.min(data["TIME"])
        last_time = start_time + so_chu_ky_train - 1

        self.__TRAIN_DATA = data[(data["TIME"] >= start_time) & (data["TIME"] <= last_time)]
        self.__TEST_DATA = data[data["TIME"] == last_time + 1]

        if self.__TRAIN_DATA.shape[0] == 0 or self.__TEST_DATA.shape[0] == 0:
            raise Exception("Dữ liệu để sinh hoặc dữ liệu để thử công thức đang bị rỗng. Kiểm tra lại số chu kỳ muốn dùng để sinh công thức.")

        self.__PROFIT = np.array(self.__TRAIN_DATA["PROFIT"], dtype=np.float64)
        self.__TEST_PROFIT = np.array(self.__TEST_DATA["PROFIT"], dtype=np.float64)

        self.__OPERAND = np.transpose(np.array(self.__TRAIN_DATA.drop(columns=drop_cols), dtype=np.float64))
        self.__TEST_OPERAND = np.transpose(np.array(self.__TEST_DATA.drop(columns=drop_cols), dtype=np.float64))

        self.__path = path_save_formula

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
    def path(self, path_save_formula):
        if type(path_save_formula) != str or not os.path.exists(path_save_formula):
            raise Exception(f'Không tồn tại thư mục {path_save_formula}/.')
        else:
            if not path_save_formula.endswith("/") and not path_save_formula.endswith("\\"):
                path_save_formula += "/"
                print(Fore.LIGHTBLUE_EX + f'Một dấu "/" đã được tự động thêm vào đường dẫn tới thư mục lưu công thức. Đường dẫn mới là: {path_save_formula}.\n', Style.RESET_ALL)

            self.__path = path_save_formula


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
        '''
        Chấp nhận cả công thức dạng array và công thức dạng string.
        '''
        if type(formula) == str:
            formula = self.convert_str_to_formula(formula)

        return _get_profit_by_weight(_calculate_formula(formula, self.__OPERAND), self.__PROFIT, self.__INDEX)


    def explain_formula(self, formula):
        '''
        Đưa ra danh mục đầu tư từng quý của công thức đầu vào.
        '''
        if type(formula) == str:
            formula = self.convert_str_to_formula(formula)

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


    def convert_npy_file_to_DataFrame(self, path_or_2d_formula_array):
        '''
        Chuyển file .npy (hoặc mảng 2 chiều gồm các công thức dưới dạng array) thành DataFrame.
        '''
        if type(path_or_2d_formula_array) == str:
            list_formula = np.load(path_or_2d_formula_array, allow_pickle=True)
        else:
            list_formula = path_or_2d_formula_array

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
