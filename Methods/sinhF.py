import numpy as np
from Methods.base import Base, convert_arrF_to_strF, decode_formula
import queryFuncs as qf
import time
import pandas as pd
import os
from Methods.bruteforceBase import set_up
import getValueFuncs as gvf
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class SinhF(Base):
    def __init__(self,
                 DB_PATH,
                 SAVE_TYPE,
                 DATA,
                 LABEL,
                 INTEREST,
                 MAX_CYC,
                 MIN_CYC,
                 LIST_FUNC,
                 DIV_WGT_BY_MC,
                 PERIODIC_SAVE_TIME=1800):
        METHOD = 2
        data, connection, list_field, main_folder, MARKET_CAP = set_up(DB_PATH, SAVE_TYPE, DATA, LABEL, MAX_CYC, MIN_CYC, METHOD, LIST_FUNC, DIV_WGT_BY_MC)
        super().__init__(data, INTEREST)
        self.connection = connection
        self.scoring_func = LIST_FUNC[0]
        self.list_field = list_field
        self.main_folder = main_folder
        self.MARKET_CAP = MARKET_CAP

        self.save_type = SAVE_TYPE
        self.periodic_save_time = PERIODIC_SAVE_TIME
        self.num_data_operand = len(self.operand_name.keys())
        self.max_cycle = MAX_CYC
        self.start_time = time.time()
        if self.save_type == 0:
            self.cursor = connection.cursor()

        self.list_data = [[] for _ in range(100)]
        self.count_n_temp = 0
        self.VALUEARG_THRESHOLD = 0.0

    def generate(self):
        # Checkpoint
        try:
            self.checkpoint = np.load(
                self.main_folder + "/checkpoint.npy", allow_pickle=True
            )
        except:
            self.checkpoint = np.array([], dtype=int)

        weight = np.zeros(len(self.PROFIT))
        self.len_checkpoint = self.checkpoint.shape[0]
        self.history = self.checkpoint.copy()

        self.list_score_F0 = np.zeros(self.num_data_operand)
        for i in range(self.num_data_operand):
            temp_weight = self.OPERAND[i]
            temp_weight[np.isnan(temp_weight)] = -1.7976931348623157e+308
            temp_weight[np.isinf(temp_weight)] = -1.7976931348623157e+308
            if self.scoring_func == "sinhF_multi_investment_strictly":
                score_F0 = gvf.sinhF_multi_investment_strictly(
                    temp_weight, self.INDEX, self.PROFIT, self.SYMBOL, self.INTEREST, self.VALUEARG, self.VALUEARG_THRESHOLD
                )

            self.list_score_F0[i] = score_F0

        self.__fill_gm2__(np.array([], dtype=int), 0, weight, -1.7976931348623157e+308)
        self.save()

    def __fill_gm2__(self, f_:np.ndarray, gen:int, w_:np.ndarray, score:float):
        if gen == 100:
            return

        if time.time() - self.start_time >= self.periodic_save_time or self.count_n_temp >= 100000:
            self.history = f_
            self.save()
            self.start_time = time.time()
            self.count_n_temp = 0
            print("Da luu", end=" ")

        if self.len_checkpoint > gen and (self.checkpoint[:gen] == f_[:gen]).all():
            start = self.checkpoint[gen]
        else:
            start = 0

        formula = np.append(f_, start)

        if gen == 0:
            stop = 2*self.num_data_operand
        else:
            stop = 4*self.num_data_operand

        sub_list = []
        if gen > 0:
            pre = formula[gen-1]
            if pre < 2*self.num_data_operand:
                sub_list.append(
                    (pre+self.num_data_operand)%(2*self.num_data_operand)
                )
                for i in range(gen-2, -1, -1):
                    ele = formula[i]
                    if ele < 2*self.num_data_operand:
                        sub_list.append(
                            (ele+self.num_data_operand)%(2*self.num_data_operand)
                        )
                    else:
                        break
            else:
                sub_list.append(
                    (pre+self.num_data_operand)%(2*self.num_data_operand)
                    + 2*self.num_data_operand
                )
                for i in range(gen-2, -1, -1):
                    ele = formula[i]
                    if ele >= 2*self.num_data_operand:
                        sub_list.append(
                            (ele+self.num_data_operand)%(2*self.num_data_operand)
                            + 2*self.num_data_operand
                        )
                    else:
                        break

        for k in range(start, stop):
            if k in sub_list:
                continue

            formula[gen] = k
            operator = k // self.num_data_operand
            operand = k % self.num_data_operand
            if operator == 0:
                weight = w_ + self.OPERAND[operand]
            elif operator == 1:
                weight = w_ - self.OPERAND[operand]
            elif operator == 2:
                weight = w_ * self.OPERAND[operand]
            else:
                weight = w_ / self.OPERAND[operand]

            weight_ = weight.copy()
            weight_[np.isnan(weight_)] = -1.7976931348623157e+308
            weight_[np.isinf(weight_)] = -1.7976931348623157e+308
            if self.scoring_func == "sinhF_multi_investment_strictly":
                cur_scr = gvf.sinhF_multi_investment_strictly(
                    weight_, self.INDEX, self.PROFIT, self.SYMBOL, self.INTEREST, self.VALUEARG, self.VALUEARG_THRESHOLD
                )
            if cur_scr > max(score, self.list_score_F0[operand]) or gen == 0:
                self.list_data[gen].append(
                    list(formula[:gen+1].copy())+[cur_scr]
                )
                self.count_n_temp += 1
                self.__fill_gm2__(formula, gen+1, weight, cur_scr)

        self.history = formula

    def save(self):
        if self.save_type == 0:
            self.cursor.execute(qf.get_list_table())
            list_table = [t_[0] for t_ in self.cursor.fetchall()]

        for k in range(100):
            if len(self.list_data[k]) > 0:
                if self.save_type == 0:
                    if f"{self.max_cycle}_{k+1}" not in list_table:
                        self.cursor.execute(qf.create_table_sinhF(k+1, self.list_field, self.max_cycle))
                        self.connection.commit()
                else:
                    list_col = ["formula"] + [self.list_field[0][0]]
                    os.makedirs(self.main_folder + f"/Gen_{k+1}", exist_ok=True)

                if self.save_type == 0:
                    self.cursor.execute(qf.insert_rows(
                        f"{self.max_cycle}_{k+1}",
                        self.list_data[k]
                    ))
                else:
                    temp_data = []
                    for lst_ in self.list_data[k]:
                        fml = convert_arrF_to_strF(decode_formula(
                            np.array(lst_[:-1], int), self.num_data_operand
                        ).astype(int))
                        temp_data.append((fml, lst_[-1]))

                    data = pd.DataFrame(temp_data)
                    data.columns = list_col
                    data.to_csv(self.main_folder + f"/Gen_{k+1}/result_{self.start_time}.csv", index=False)

                self.list_data[k].clear()

        np.save(self.main_folder + "/checkpoint.npy", self.history, allow_pickle=True)
        if self.save_type == 0:
            self.connection.commit()
        self.start_time = time.time()
        print("Da luu")
