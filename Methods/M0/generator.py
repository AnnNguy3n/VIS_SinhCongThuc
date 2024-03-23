from Methods.bruteforceBase import BruteforceBase
import numpy as np
import copy
from Methods.M0.helpFuncs import *
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class Generator(BruteforceBase):
    def __init__(self,
                 DB_PATH,
                 SAVE_TYPE,
                 DATA,
                 LABEL,
                 INTEREST,
                 VALUEARG_THRESHOLD,
                 NUM_CYC,
                 MAX_CYC,
                 MIN_CYC,
                 LIST_FUNC,
                 NUM_PROCESS,
                 FILTERS,
                 DIV_WGT_BY_MC,
                 TARGET,
                 TMP_STRG_SIZE=10000,
                 PERIODIC_SAVE_TIME=1800,
                 MAX_OPR_PER_FML=100,
                 numerator_condition=True):
        METHOD = 0
        super().__init__(DB_PATH, SAVE_TYPE, DATA, LABEL, INTEREST, VALUEARG_THRESHOLD, NUM_CYC, MAX_CYC, MIN_CYC, METHOD, LIST_FUNC, NUM_PROCESS, FILTERS, DIV_WGT_BY_MC, TARGET, TMP_STRG_SIZE, PERIODIC_SAVE_TIME, MAX_OPR_PER_FML)
        self.numerator_condition = numerator_condition

    def generate(self, list_required_operand=[]):
        # Checkpoint
        try:
            self.checkpoint = list(np.load(
                self.main_folder + "/checkpoint.npy", allow_pickle=True
            ))
            self.checkpoint[0][-1] += 1
            self.checkpoint[-1][-1] += 1
        except:
            self.checkpoint = [
                np.array([
                    max(len(list_required_operand), 1), # Do dai cong thuc
                    0, # So toan hang trong cac cum tru
                    0, # Index cua cau truc cum cong
                    0, # Index cua cau truc cum tru,
                    0 # id
                ]),
                None, # Cau truc cong thuc
                None # Cong thuc cuoi cung sinh den
            ]

        self.start_id = self.checkpoint[0][-1]

        # Kiem tra va tao bang
        num_operand = self.checkpoint[0][0]
        self.current_formula_length = num_operand
        self.check_and_create_table(num_operand)

        # Generate
        self.current = copy.deepcopy(self.checkpoint)
        self.count = np.array([0, self.storage_size, 0, 1000000000])
        list_required_operand = np.array(list_required_operand, int)
        if num_operand > self.max_operand_per_formula:
            return

        while True:
            self.current[0][0] = num_operand

            start_num_sub_operand = 0
            if self.current[0][0] == self.checkpoint[0][0]:
                start_num_sub_operand = self.checkpoint[0][1]

            for num_sub_operand in range(start_num_sub_operand, num_operand+1):
                self.current[0][1] = num_sub_operand
                temp_arr = np.full(num_sub_operand, 0)
                list_sub_struct = list([temp_arr])
                list_sub_struct.pop(0)
                split_posint_into_sum(num_sub_operand, temp_arr, list_sub_struct)

                num_add_operand = num_operand - num_sub_operand
                temp_arr = np.full(num_add_operand, 0)
                list_add_struct = list([temp_arr])
                list_add_struct.pop(0)
                split_posint_into_sum(num_add_operand, temp_arr, list_add_struct)

                start_add_struct_idx = 0
                if (self.current[0][0:2] == self.checkpoint[0][0:2]).all():
                    start_add_struct_idx = self.checkpoint[0][2]

                for add_struct_idx in range(start_add_struct_idx, len(list_add_struct)):
                    self.current[0][2] = add_struct_idx

                    start_sub_struct_idx =  0
                    if (self.current[0][0:3] == self.checkpoint[0][0:3]).all():
                        start_sub_struct_idx = self.checkpoint[0][3]

                    for sub_struct_idx in range(start_sub_struct_idx, len(list_sub_struct)):
                        self.current[0][3] = sub_struct_idx
                        add_struct = list_add_struct[add_struct_idx][list_add_struct[add_struct_idx]>0]
                        sub_struct = list_sub_struct[sub_struct_idx][list_sub_struct[sub_struct_idx]>0]
                        if self.checkpoint[1] is None:
                            struct = create_struct(add_struct, sub_struct)
                            self.checkpoint[1] = struct
                        elif (self.current[0][:4] == self.checkpoint[0][:4]).all():
                            struct = self.checkpoint[1].copy()
                        else: struct = create_struct(add_struct, sub_struct)

                        self.current[1] = struct.copy()

                        while True:
                            if self.checkpoint[2] is None:
                                formula = create_formula(struct)
                                self.checkpoint[2] = formula
                            elif struct.shape == self.checkpoint[1].shape and (struct == self.checkpoint[1]).all() and (self.current[0][:4] == self.checkpoint[0][:4]).all():
                                formula = self.checkpoint[2].copy()
                            else: formula = create_formula(struct)

                            self.current[2] = formula.copy()
                            if list_required_operand.shape[0] == formula.shape[0] // 2:
                                sub_mode = True
                            else:
                                sub_mode = False

                            self.__fill_gm0__(formula, struct, 1, np.zeros(self.OPERAND.shape[1]), -1, np.zeros(self.OPERAND.shape[1]), sub_mode, list_required_operand)
                            if not update_struct(struct, self.numerator_condition):
                                break

            # self.save_history(flag=0)

            num_operand += 1
            if num_operand == self.max_operand_per_formula + 1:
                self.save_history(flag=1)
                return
            else:
                self.save_history(flag=0)

            self.current_formula_length = num_operand
            self.start_id = 0
            self.check_and_create_table(num_operand)

    def __fill_gm0__(self, formula, struct, idx, temp_0, temp_op, temp_1, sub_mode, list_op):
        start = 0
        if (formula[0:idx] == self.current[2][0:idx]).all():
            start = self.current[2][idx]

        valid_operand = get_valid_operand(formula, struct, idx, start, self.OPERAND.shape[0])
        if sub_mode:
            valid_operand = np.intersect1d(valid_operand, list_op)

        if valid_operand.shape[0] > 0:
            if formula[idx-1] < 2:
                temp_op_new = formula[idx-1]
                temp_1_new = self.OPERAND[valid_operand]
            else:
                temp_op_new = temp_op
                if formula[idx-1] == 2:
                    temp_1_new = temp_1 * self.OPERAND[valid_operand]
                else:
                    temp_1_new = temp_1 / self.OPERAND[valid_operand]

            if idx + 1 == formula.shape[0] or formula[idx+1] < 2:
                if temp_op_new == 0:
                    temp_0_new = temp_0 + temp_1_new
                else:
                    temp_0_new = temp_0 - temp_1_new
            else:
                temp_0_new = np.array([temp_0]*valid_operand.shape[0])

            if idx + 1 != formula.shape[0]:
                temp_list_formula = np.array([formula]*valid_operand.shape[0])
                temp_list_formula[:,idx] = valid_operand
                idx_new = idx + 2
                for i in range(valid_operand.shape[0]):
                    if valid_operand[i] in list_op:
                        new_list_op = list_op[list_op != valid_operand[i]]
                        new_sub_mode = sub_mode
                    else:
                        new_list_op = list_op
                        if idx + 1 + 2*list_op.shape[0] == formula.shape[0]:
                            new_sub_mode = True
                        else:
                            new_sub_mode = sub_mode

                    self.__fill_gm0__(temp_list_formula[i], struct, idx_new, temp_0_new[i], temp_op_new, temp_1_new[i], new_sub_mode, new_list_op)
            else:
                temp_0_new[np.isnan(temp_0_new)] = -1.7976931348623157e+308
                temp_0_new[np.isinf(temp_0_new)] = -1.7976931348623157e+308

                formulas = np.array([formula]*valid_operand.shape[0])
                formulas[:, idx] = valid_operand

                self.count[0:3:2] += self.add_to_temp_storage(temp_0_new, formulas)
                self.current[2][:] = formula[:]
                self.current[2][idx] = self.num_data_operand

                if self.count[0] >= self.count[1] or self.count[2] >= self.count[3]:
                    self.save_history(flag=0)

    def change_checkpoint(self):
        self.current[0][-1] = self.start_id - 1
