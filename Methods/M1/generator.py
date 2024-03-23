from Methods.bruteforceBase import BruteforceBase
import numpy as np
import copy
from Methods.M0.helpFuncs import get_valid_operand
from Methods.M1.helpFuncs import get_valid_op
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class Generator(BruteforceBase):
    def __init__(self,
                 DB_PATH,
                 SAVE_TYPE,
                 DATA,
                 LABEL,
                 INTEREST,
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
                 MAX_OPR_PER_FML=100):
        METHOD = 1
        super().__init__(DB_PATH, SAVE_TYPE, DATA, LABEL, INTEREST, NUM_CYC, MAX_CYC, MIN_CYC, METHOD, LIST_FUNC, NUM_PROCESS, FILTERS, DIV_WGT_BY_MC, TARGET, TMP_STRG_SIZE, PERIODIC_SAVE_TIME, MAX_OPR_PER_FML)

    def generate(self, list_required_operand=[]):
        # Checkpoint
        try:
            self.checkpoint = list(np.load(
                self.main_folder + "/checkpoint.npy", allow_pickle=True
            ))
            self.checkpoint[-1] += 1
            self.checkpoint[0][-1] += 1
        except:
            self.checkpoint = [
                np.zeros(2*max(len(list_required_operand), 1), int), 0, 0
            ]

        self.start_id = self.checkpoint[-1]

        # Kiem tra va tao bang
        num_operand = self.checkpoint[0].shape[0] // 2
        self.current_formula_length = num_operand
        self.check_and_create_table(num_operand)

        # Generate
        self.current = copy.deepcopy(self.checkpoint)
        self.count = np.array([0, self.storage_size, 0, 1000000000])
        last_operand = num_operand
        list_required_operand = np.array(list_required_operand, int)

        while True:
            list_uoc_so = [i for i in range(1, num_operand+1) if num_operand % i == 0]
            start_divisor_idx = 0
            if num_operand == last_operand:
                start_divisor_idx = self.checkpoint[1]

            formula = np.full(num_operand*2, 0)
            for i in range(start_divisor_idx, len(list_uoc_so)):
                struct = np.array([[0, list_uoc_so[i], 1+2*list_uoc_so[i]*j, 0] for j in range(num_operand//list_uoc_so[i])])
                if num_operand != last_operand or i != self.current[1]:
                    self.current[0] = formula.copy()
                    self.current[1] = i

                if list_required_operand.shape[0] == formula.shape[0] // 2:
                    sub_mode = True
                else:
                    sub_mode = False

                self.__fill_gm1__(formula, struct, 0, np.zeros(self.OPERAND.shape[1]), 0, np.zeros(self.OPERAND.shape[1]), 0, False, False, sub_mode, list_required_operand)

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

    def __fill_gm1__(self, formula, struct, idx, temp_0, temp_op, temp_1, mode, add_sub_done, mul_div_done, sub_mode, list_op):
        if mode == 0: # Sinh dấu cộng trừ đầu mỗi cụm
            gr_idx = list(struct[:,2]-1).index(idx)

            start = 0
            if (formula[0:idx] == self.current[0][0:idx]).all():
                start = self.current[0][idx]

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

                self.__fill_gm1__(new_formula, new_struct, idx+1, temp_0, temp_op, temp_1, 1, new_add_sub_done, mul_div_done, sub_mode, list_op)
        elif mode == 2:
            start = 2
            if (formula[0:idx] == self.current[0][0:idx]).all():
                start = self.current[0][idx]

            if start == 0:
                start = 2

            valid_op = get_valid_op(struct, idx, start)
            for op in valid_op:
                new_formula = formula.copy()
                new_struct = struct.copy()
                new_formula[idx] = op
                if op == 3:
                    new_mul_div_done = True
                    for i in range(idx+2, 2*new_struct[0,1]-1, 2):
                        new_formula[i] = 3

                    for i in range(1, new_struct.shape[0]):
                        for j in range(new_struct[0,1]-1):
                            new_formula[new_struct[i,2] + 2*j + 1] = new_formula[2+2*j]
                else:
                    new_struct[:,3] += 1
                    new_mul_div_done = False
                    if idx == 2*new_struct[0,1] - 2:
                        new_mul_div_done = True
                        for i in range(1, new_struct.shape[0]):
                            for j in range(new_struct[0,1]-1):
                                new_formula[new_struct[i,2] + 2*j + 1] = new_formula[2+2*j]

                self.__fill_gm1__(new_formula, new_struct, idx+1, temp_0, temp_op, temp_1, 1, add_sub_done, new_mul_div_done, sub_mode, list_op)
        elif mode == 1:
            start = 0
            if (formula[0:idx] == self.current[0][0:idx]).all():
                start = self.current[0][idx]

            valid_operand = get_valid_operand(formula, struct, idx, start, self.OPERAND.shape[0])
            if sub_mode:
                valid_operand = np.intersect1d(valid_operand, list_op)

            if valid_operand.shape[0] > 0:
                if formula[idx-1] < 2:
                    temp_op_new = formula[idx-1]
                    temp_1_new = self.OPERAND[valid_operand].copy()
                else:
                    temp_op_new = temp_op
                    if formula[idx-1] == 2:
                        temp_1_new = temp_1 * self.OPERAND[valid_operand]
                    else:
                        temp_1_new = temp_1 / self.OPERAND[valid_operand]

                if idx + 1 == formula.shape[0] or (idx+2) in struct[:,2]:
                    if temp_op_new == 0:
                        temp_0_new = temp_0 + temp_1_new
                    else:
                        temp_0_new = temp_0 - temp_1_new
                else:
                    temp_0_new = np.array([temp_0]*valid_operand.shape[0])

                if idx + 1 != formula.shape[0]:
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
                        if valid_operand[i] in list_op:
                            new_list_op = list_op[list_op != valid_operand[i]]
                            new_sub_mode = sub_mode
                        else:
                            new_list_op = list_op.copy()
                            if idx + 1 + 2*list_op.shape[0] == formula.shape[0]:
                                new_sub_mode = True
                            else:
                                new_sub_mode = sub_mode

                        self.__fill_gm1__(temp_list_formula[i], struct, new_idx, temp_0_new[i], temp_op_new, temp_1_new[i], new_mode, add_sub_done, mul_div_done, new_sub_mode, new_list_op)
                else:
                    temp_0_new[np.isnan(temp_0_new)] = -1.7976931348623157e+308
                    temp_0_new[np.isinf(temp_0_new)] = -1.7976931348623157e+308

                    formulas = np.array([formula]*valid_operand.shape[0])
                    formulas[:, idx] = valid_operand

                    self.count[0:3:2] += self.add_to_temp_storage(temp_0_new, formulas)
                    self.current[0][:] = formula[:]
                    self.current[0][idx] = self.OPERAND.shape[0]

                    if self.count[0] >= self.count[1] or self.count[2] >= self.count[3]:
                        self.save_history(flag=0)

    def change_checkpoint(self):
        self.current[-1] = self.start_id - 1
