from base import Method, _get_profit_by_weight
import pandas as pd
import numpy as np
import os
from numba import njit
from numba.typed import List
from colorama import Fore, Style
import copy
from datetime import datetime


@njit
def _get_valid_operand(formula, struct, idx, start, num_operand):
    valid_operand = np.full(num_operand, 0)
    valid_operand[start:num_operand] = 1

    for i in range(struct.shape[0]):
        if struct[i,2] + 2*struct[i,1] > idx:
            gr_idx = i
            break

    """
    Tránh hoán vị nhân chia trong một cụm
    """
    pre_op = formula[idx-1]
    if pre_op >= 2:
        if pre_op == 2:
            temp_idx = struct[gr_idx,2]
            if idx >= temp_idx + 2:
                valid_operand[0:formula[idx-2]] = 0
        else:
            temp_idx = struct[gr_idx,2]
            temp_idx_1 = temp_idx + 2*struct[gr_idx,3]
            if idx > temp_idx_1 + 2:
                valid_operand[0:formula[idx-2]] = 0

            """
            Tránh chia lại những toán hạng đã nhân ở trong cụm (chỉ phép chia mới check)
            """
            valid_operand[formula[temp_idx:temp_idx_1+1:2]] = 0

    """
    Tránh hoán vị cộng trừ các cụm, kể từ cụm thứ 2 trở đi
    """
    if gr_idx > 0:
        gr_check_idx = -1
        for i in range(gr_idx-1,-1,-1):
            if struct[i,0]==struct[gr_idx,0] and struct[i,1]==struct[gr_idx,1] and struct[i,3]==struct[gr_idx,3]:
                gr_check_idx = i
                break

        if gr_check_idx != -1:
            idx_ = 0
            while True:
                idx_1 = struct[gr_idx,2] + idx_
                idx_2 = struct[gr_check_idx,2] + idx_
                if idx_1 == idx:
                    valid_operand[0:formula[idx_2]] = 0
                    break

                if formula[idx_1] != formula[idx_2]:
                    break

                idx_ += 2

        """
        Tránh trừ đi những cụm đã cộng trước đó (chỉ ở trong trừ cụm mới check)
        """
        if struct[gr_idx,0] == 1 and idx + 2 == struct[gr_idx,2] + 2*struct[gr_idx,1]:
            list_gr_check = np.where((struct[:,0]==0) & (struct[:,1]==struct[gr_idx,1]) & (struct[:,3]==struct[gr_idx,3]))[0]
            for i in list_gr_check:
                temp_idx = struct[i,2] + 2*struct[i,1] - 2
                temp_idx_1 = struct[gr_idx,2] + 2*struct[gr_idx,1] - 2
                if (formula[struct[i,2]:temp_idx] == formula[struct[gr_idx,2]:temp_idx_1]).all():
                    valid_operand[formula[temp_idx]] = 0

    return np.where(valid_operand==1)[0]


@njit
def _get_profits_by_weights(weights, profit, index):
    result = np.zeros(weights.shape[0])
    for i in range(weights.shape[0]):
        result[i] = _get_profit_by_weight(weights[i], profit, index)

    return result


@njit
def _split_posint_into_sum(n, arr, list_result):
    if np.sum(arr) == n:
        list_result.append(arr)
    else:
        idx = np.where(arr==0)[0][0]
        sum_ = np.sum(arr)
        if idx == 0:
            max_ = n
        else:
            max_ = arr[idx-1]

        max_ = min(n-sum_, max_)
        for i in range(max_, 0, -1):
            arr[idx] = i
            _split_posint_into_sum(n, arr.copy(), list_result)


@njit
def _create_struct(add_struct, sub_struct):
    struct = np.full((add_struct.shape[0]+sub_struct.shape[0], 4), -1)
    temp_val = 1
    for i in range(add_struct.shape[0]):
        struct[i,:] = np.array([0, add_struct[i], temp_val, add_struct[i]-1])
        temp_val += 2*struct[i,1]

    for i in range(sub_struct.shape[0]):
        temp_val_1 = add_struct.shape[0] + i
        struct[temp_val_1,:] = np.array([1, sub_struct[i], temp_val, sub_struct[i]-1])
        temp_val += 2*struct[temp_val_1,1]

    return struct


@njit
def _create_formula(struct):
    n = np.sum(struct[:,1])
    formula = np.full(2*n, 0)
    temp_val = 0
    for i in range(struct.shape[0]):
        temp = struct[i]
        formula[temp_val] = temp[0]
        temp_val += 2
        for j in range(temp[1]-1):
            if j < temp[3]:
                formula[temp_val] = 2
            else:
                formula[temp_val] = 3

            temp_val += 2

    return formula


@njit
def _update_struct(struct):
    for i in range(struct.shape[0]-1, -1, -1):
        if struct[i,3] > 0:
            temp = np.where((struct[i:,0]==struct[i,0]) & (struct[i:,1]==struct[i,1]))[0] + i
            struct[temp,3] = struct[i,3] - 1
            temp_1 = np.max(temp) + 1
            struct[temp_1:,3] = struct[temp_1:,1] - 1
            return True

    return False


@njit
def _njit_fill_operand(formula, struct, idx, temp_0, temp_op, temp_1, target, last_formula, operand, profit, index, list_formula, count):
        start = -1
        if (formula[0:idx]==last_formula[0:idx]).all():
            start = last_formula[idx]
        else:
            start = 0

        valid_operand = _get_valid_operand(formula, struct, idx, start, operand.shape[0])
        if valid_operand.shape[0] > 0:
            if formula[idx-1] < 2:
                temp_op_new = formula[idx-1]
                temp_1_new = operand[valid_operand].copy()
            else:
                temp_op_new = temp_op
                if formula[idx-1] == 2:
                    temp_1_new = temp_1 * operand[valid_operand]
                else:
                    temp_1_new = temp_1 / operand[valid_operand]

            if idx + 1 == formula.shape[0] or formula[idx+1] < 2:
                if temp_op_new == 0:
                    temp_0_new = temp_0 + temp_1_new
                else:
                    temp_0_new = temp_0 - temp_1_new
            else:
                temp_0_new = np.zeros((valid_operand.shape[0], temp_0.shape[0])) + temp_0

            if idx + 1 == formula.shape[0]:
                for arr in temp_0_new:
                    arr[np.isnan(arr)] = -1.7976931348623157e+308
                    arr[np.isinf(arr)] = -1.7976931348623157e+308

                temp_profits = _get_profits_by_weights(temp_0_new, profit, index)
                valid_formula = np.where(temp_profits>=target)[0]
                if valid_formula.shape[0] > 0:
                    temp_list_formula = np.full((valid_formula.shape[0], formula.shape[0]), 0) + formula
                    temp_list_formula[:,idx] = valid_operand[valid_formula]
                    list_formula[count[0]:count[0]+valid_formula.shape[0]] = temp_list_formula
                    count[0:3:2] += valid_formula.shape[0]

                last_formula[:] = formula[:]
                last_formula[idx] = operand.shape[0]

                if count[0] >= count[1] or count[2] >= count[3]:
                    return True
            else:
                temp_list_formula = np.full((valid_operand.shape[0], formula.shape[0]), 0) + formula
                temp_list_formula[:,idx] = valid_operand
                idx_new = idx + 2
                for i in range(valid_operand.shape[0]):
                    if _njit_fill_operand(temp_list_formula[i], struct, idx_new, temp_0_new[i], temp_op_new, temp_1_new[i], target, last_formula, operand, profit, index, list_formula, count):
                        return True

        return False



class CompleteMethod(Method):
    def __init__(self, data: pd.DataFrame, pathSaveFormula: str, soChuKyTrain: int) -> None:
        super().__init__(data, pathSaveFormula, soChuKyTrain)


    def readMe(self):
        print("Sinh vét cạn.")


    def __fill_operand(self, formula, struct, idx, temp_0, temp_op, temp_1, target):
        start = -1
        if (formula[0:idx]==self.__current[5][0:idx]).all():
            start = self.__current[5][idx]
        else:
            start = 0

        valid_operand = _get_valid_operand(formula, struct, idx, start, self._Method__OPERAND.shape[0])
        if valid_operand.shape[0] > 0:
            if formula[idx-1] < 2:
                temp_op_new = formula[idx-1]
                temp_1_new = self._Method__OPERAND[valid_operand].copy()
            else:
                temp_op_new = temp_op
                if formula[idx-1] == 2:
                    temp_1_new = temp_1 * self._Method__OPERAND[valid_operand]
                else:
                    temp_1_new = temp_1 / self._Method__OPERAND[valid_operand]

            if idx + 1 == formula.shape[0] or formula[idx+1] < 2:
                if temp_op_new == 0:
                    temp_0_new = temp_0 + temp_1_new
                else:
                    temp_0_new = temp_0 - temp_1_new
            else:
                temp_0_new = np.array([temp_0]*valid_operand.shape[0])

            if idx + 1 == formula.shape[0]:
                temp_0_new[np.isnan(temp_0_new)] = -1.7976931348623157e+308
                temp_0_new[np.isinf(temp_0_new)] = -1.7976931348623157e+308
                temp_profits = _get_profits_by_weights(temp_0_new, self._Method__PROFIT, self._Method__INDEX)
                valid_formula = np.where(temp_profits>=target)[0]
                if valid_formula.shape[0] > 0:
                    temp_list_formula = np.array([formula]*valid_formula.shape[0])
                    temp_list_formula[:,idx] = valid_operand[valid_formula]
                    self.__list_formula[self.__count[0]:self.__count[0]+valid_formula.shape[0]] = temp_list_formula
                    self.__count[0:3:2] += valid_formula.shape[0]

                self.__current[5][:] = formula[:]
                self.__current[5][idx] = self._Method__OPERAND.shape[0]

                if self.__count[0] >= self.__count[1] or self.__count[2] >= self.__count[3]:
                    return True
            else:
                temp_list_formula = np.array([formula]*valid_operand.shape[0])
                temp_list_formula[:,idx] = valid_operand
                idx_new = idx + 2
                if formula.shape[0] - 7 <= idx_new:
                    for i in range(valid_operand.shape[0]):
                        if _njit_fill_operand(temp_list_formula[i], struct, idx_new, temp_0_new[i], temp_op_new, temp_1_new[i], target, self.__current[5], self._Method__OPERAND, self._Method__PROFIT, self._Method__INDEX, self.__list_formula, self.__count):
                            return True
                else:
                    for i in range(valid_operand.shape[0]):
                        if self.__fill_operand(temp_list_formula[i], struct, idx_new, temp_0_new[i], temp_op_new, temp_1_new[i], target):
                            return True

        return False


    def generate_formula(self, target_profit=1.0, formula_file_size=1000000, target_num_formula=1000000000):
        '''
        * target_profit: Lợi nhuận mong muốn.
        * formula_file_size: Số lượng công thức xấp xỉ trong mỗi file lưu trữ.
        * target_num_formula: Số công thức đạt điều kiện được sinh trong 1 lần chạy ko ngắt.
        '''
        print(Fore.LIGHTYELLOW_EX+"Khi ngắt bằng tay thì cần tự chạy phương thức <CompleteMethod_object>.save_history() để lưu lịch sử.", Style.RESET_ALL)

        try:
            temp = np.load(self.path+"history.npy", allow_pickle=True)
            self.__history = temp
        except:
            self.__history = [
                1, # Số toán hạng có trong công thức
                0, # Số toán hạng trong các trừ cụm
                0, # Cấu trúc các cộng cụm thứ mấy
                0, # Cấu trúc các trừ cụm thứ mấy
                np.array([[0, 1, 1, 0]]), # Cấu trúc công thức tổng quát
                np.array([0, 0]) # Công thức đã sinh đến trong lịch sử
            ]

        self.__current = copy.deepcopy(self.__history)

        self.__count = np.array([0, formula_file_size, 0, target_num_formula])

        num_operand = self.__history[0] - 1
        while True:
            num_operand += 1
            print("Đang chạy sinh công thức có số toán hạng là ", num_operand, ". . .")
            if self._Method__OPERAND.shape[0] <= 256:
                self.__list_formula = np.full((formula_file_size+self._Method__OPERAND.shape[0], 2*num_operand), 0, dtype=np.uint8)
            else:
                self.__list_formula = np.full((formula_file_size+self._Method__OPERAND.shape[0], 2*num_operand), 0, dtype=np.uint16)

            if num_operand == self.__history[0]:
                start_num_sub_operand = self.__history[1]
            else: start_num_sub_operand = 0

            for num_sub_operand in range(start_num_sub_operand, num_operand+1):
                temp_arr = np.full(num_sub_operand, 0)
                list_sub_struct = List([temp_arr])
                list_sub_struct.pop(0)
                _split_posint_into_sum(num_sub_operand, temp_arr, list_sub_struct)

                num_add_operand = num_operand - num_sub_operand
                temp_arr = np.full(num_add_operand, 0)
                list_add_struct = List([temp_arr])
                list_add_struct.pop(0)
                _split_posint_into_sum(num_add_operand, temp_arr, list_add_struct)

                if num_sub_operand == self.__history[1] and num_operand == self.__history[0]:
                    start_add_struct_idx = self.__history[2]
                else: start_add_struct_idx = 0

                for add_struct_idx in range(start_add_struct_idx, len(list_add_struct)):
                    if  add_struct_idx == self.__history[2] and \
                        num_sub_operand == self.__history[1] and num_operand == self.__history[0]:
                        start_sub_struct_idx = self.__history[3]
                    else: start_sub_struct_idx =  0

                    for sub_struct_idx in range(start_sub_struct_idx, len(list_sub_struct)):
                        add_struct = list_add_struct[add_struct_idx][list_add_struct[add_struct_idx]>0]
                        sub_struct = list_sub_struct[sub_struct_idx][list_sub_struct[sub_struct_idx]>0]
                        if  sub_struct_idx == self.__history[3] and add_struct_idx == self.__history[2] and \
                            num_sub_operand == self.__history[1] and num_operand == self.__history[0]:
                            struct = self.__history[4].copy()
                        else: struct = _create_struct(add_struct, sub_struct)

                        while True:
                            if struct.shape == self.__history[4].shape and (struct==self.__history[4]).all():
                                formula = self.__history[5].copy()
                            else:
                                formula = _create_formula(struct)

                            self.__current[0] = num_operand
                            self.__current[1] = num_sub_operand
                            self.__current[2] = add_struct_idx
                            self.__current[3] = sub_struct_idx
                            self.__current[4] = struct.copy()
                            self.__current[5] = formula.copy()

                            while self.__fill_operand(formula, struct, 1, np.zeros(self._Method__OPERAND.shape[1]), -1, np.zeros(self._Method__OPERAND.shape[1]), target_profit):
                                self.save_history()

                            if not _update_struct(struct):
                                break

            self.save_history()
    @property
    def current(self):
        return copy.deepcopy(self.__current)
    @property
    def count(self):
        return self.__count.copy()


    def save_history(self):
        np.save(self.path+"history.npy", self.__current)
        print(Fore.LIGHTGREEN_EX+"Đã lưu lịch sử.", Style.RESET_ALL)
        if self.__count[0] == 0:
            return

        num_operand = self.__current[0]
        while True:
            pathSave = self.path + f"high_profit_{num_operand}_" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".npy"
            if not os.path.exists(pathSave):
                np.save(pathSave, self.__list_formula[0:self.__count[0]])
                self.__count[0] = 0
                print(Fore.LIGHTGREEN_EX+"Đã lưu công thức", Style.RESET_ALL)
                if self.__count[2] >= self.__count[3]:
                    raise Exception("Đã sinh đủ công thức theo yêu cầu.")

                return
