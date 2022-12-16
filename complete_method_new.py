from base import Method
import pandas as pd
import numpy as np
import os
from numba.typed import List
from colorama import Fore, Style
import copy
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from nopy import _get_valid_operand, _get_geomean_profits_by_weights, _get_valid_op


class CompleteMethod_new(Method):
    def __init__(self, data: pd.DataFrame, path_save_formula: str, so_chu_ky_train: int, test_start: int) -> None:
        super().__init__(data, path_save_formula, so_chu_ky_train)

        self._Method__num_test = self._Method__INDEX.shape[0] - test_start + 1
    

    def __fill_operand(self, formula, struct, idx, temp_0, temp_op, temp_1, target, mode, add_sub_done, mul_div_done):
        if mode == 0: # Sinh dấu cộng trừ đầu mỗi cụm
            # Tìm idx cụm
            gr_idx = list(struct[:,2]-1).index(idx)

            start = 0
            if (formula[0:idx] == self.__last_formula[0:idx]).all():
                start = self.__last_formula[idx]
            
            for op in range(start, 2):
                new_formula = formula.copy()
                new_struct = struct.copy()
                new_formula[idx] = op
                new_struct[gr_idx,0] = op
                if op == 1:
                    new_add_sub_done = True
                    new_formula[new_struct[gr_idx+1:,2]-1] = 1
                    new_struct[gr_idx+1:,0] = 1
                else:
                    new_add_sub_done = False
                
                if self.__fill_operand(new_formula, new_struct, idx+1, temp_0, temp_op, temp_1, target, 1, new_add_sub_done, mul_div_done):
                    return True
        
        elif mode == 1:
            start = 0
            if (formula[0:idx] == self.__last_formula[0:idx]).all():
                start = self.__last_formula[idx]

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
                    valid_idx, check_target = _get_geomean_profits_by_weights(temp_0_new, self._Method__PROFIT, self._Method__INDEX, self._Method__num_test, target)
                    if valid_idx.shape[0] > 0:
                        temp_list_formula = np.array([formula]*valid_idx.shape[0])
                        temp_list_formula[:,idx] = valid_operand[valid_idx]
                        x_1 = self.__count[0]
                        x_2 = self.__count[0] + valid_idx.shape[0]
                        self.__list_formula[0][x_1:x_2] = temp_list_formula
                        self.__list_formula[1][x_1:x_2] = check_target
                        self.__count[0:3:2] += valid_idx.shape[0]
                    
                    self.__last_formula[:] = formula[:]
                    self.__last_formula[idx] = self._Method__OPERAND.shape[0]
                    if self.__count[0] >= self.__count[1] or self.__count[2] >= self.__count[3]:
                        return True
                else:
                    temp_list_formula = np.array([formula]*valid_operand.shape[0])
                    temp_list_formula[:,idx] = valid_operand
                    if idx + 2 in struct[:,2]:
                        if add_sub_done:
                            new_idx = idx + 2
                            new_mode = 1
                        else:
                            new_idx = idx + 1
                            new_mode = 0
                    else:
                        if mul_div_done:
                            new_idx = idx + 2
                            new_mode = 1
                        else:
                            new_idx = idx + 1
                            new_mode = 2
                    
                    for i in range(valid_operand.shape[0]):
                        if self.__fill_operand(temp_list_formula[i], struct, new_idx, temp_0_new[i], temp_op_new, temp_1_new[i], target, new_mode, add_sub_done, mul_div_done):
                            return True

        elif mode == 2:
            start = 2
            if (formula[0:idx] == self.__last_formula[0:idx]).all():
                start = self.__last_formula[idx]

            if start == 0:
                start = 2
            
            valid_op = _get_valid_op(formula, struct, idx, start)
            for op in valid_op:
                new_formula = formula.copy()
                new_struct = struct.copy()
                new_formula[idx] = op
                if op == 3:
                    new_mul_div_done = True
                    for i in range(idx+2, 2*struct[0,1]-1, 2):
                        new_formula[i] = 3
                    
                    for i in range(1, struct.shape[0]):
                        for j in range(struct[0,1]-1):
                            new_formula[struct[i,2] + 2*j] = new_formula[1+2*j]
                else:
                    new_struct[:,3] += 1
                    new_mul_div_done = False
                    if idx == 2*new_struct[0,1] - 2:
                        new_mul_div_done = True
                        for i in range(1, struct.shape[0]):
                            for j in range(struct[0,1]-1):
                                new_formula[struct[i,2] + 2*j] = new_formula[1+2*j]
                
                if self.__fill_operand(new_formula, new_struct, idx+1, temp_0, temp_op, temp_1, target, 1, add_sub_done, new_mul_div_done):
                    return True
        
        return False
    

    def generate_formula(self, target_profit=1.0, formula_file_size=1000000, target_num_formula=1000000000):
        print(Fore.LIGHTYELLOW_EX+"Khi ngắt bằng tay thì cần tự chạy phương thức <CompleteMethod_object>.save_history() để lưu lịch sử.", Style.RESET_ALL)

        try:
            temp = np.load(self.path+"history_new.npy", allow_pickle=True)
            self.__history = temp
        except:
            self.__history =  np.array([0, 0]), 0
        
        self.__last_formula = self.__history[0].copy()
        self.__last_uoc_idx = self.__history[1]

        self.__count = np.array([0, formula_file_size, 0, target_num_formula])

        last_operand = self.__last_formula.shape[0] // 2
        num_operand = last_operand - 1

        while True:
            num_operand += 1
            print("Đang chạy sinh công thức có số toán hạng là ", num_operand, ". . .")

            if self._Method__OPERAND.shape[0] <= 256:
                self.__list_formula = [np.full((formula_file_size+self._Method__OPERAND.shape[0], 2*num_operand), 0, dtype=np.uint8), np.full((formula_file_size+self._Method__OPERAND.shape[0], self._Method__num_test), 0, dtype=np.uint8), -1]
            else:
                self.__list_formula = [np.full((formula_file_size+self._Method__OPERAND.shape[0], 2*num_operand), 0, dtype=np.uint16), np.full((formula_file_size+self._Method__OPERAND.shape[0], self._Method__num_test), 0, dtype=np.uint8), -1]
            
            list_uoc_so = []
            for i in range(1, num_operand+1):
                if num_operand % i == 0:
                    list_uoc_so.append(i)
            
            start_uoc_idx = 0
            if num_operand == last_operand:
                start_uoc_idx = self.__history[1]
            
            formula = np.full(num_operand*2, 0)
            for i in range(start_uoc_idx, len(list_uoc_so)):
                print("Số phần tử trong 1 cụm", list_uoc_so[i])
                struct = np.array([[0, list_uoc_so[i], 1+2*list_uoc_so[i]*j, 0] for j in range(num_operand//list_uoc_so[i])])
                if num_operand != last_operand:
                    self.__last_formula = formula.copy()
                    self.__last_uoc_idx = i
                
                while self.__fill_operand(formula, struct, 0, np.zeros(self._Method__OPERAND.shape[1]), 0, np.zeros(self._Method__OPERAND.shape[1]), target_profit, 0, False, False):
                    self.save_history()
            
            if self.save_history():
                break
    
    def save_history(self):
        np.save(self.path+"history_new.npy", (self.__last_formula, self.__last_uoc_idx))
        print(Fore.LIGHTGREEN_EX+"Đã lưu lịch sử.", Style.RESET_ALL)
        if self.__count[0] == 0:
            return False
        
        num_operand = self.__last_formula.shape[0] // 2
        while True:
            pathSave = self.path + f"high_profit_{num_operand}_" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".npy"
            if not os.path.exists(pathSave):
                np.save(pathSave, [self.__list_formula[0][0:self.__count[0]], self.__list_formula[1][0:self.__count[0]], -1])
                self.__count[0] = 0
                print(Fore.LIGHTGREEN_EX+"Đã lưu công thức", Style.RESET_ALL)
                if self.__count[2] >= self.__count[3]:
                    raise Exception("Đã sinh đủ công thức theo yêu cầu.")
                    
                return False