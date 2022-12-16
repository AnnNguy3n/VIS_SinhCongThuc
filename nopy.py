from numba import njit
import numpy as np


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
        if formula[i] >= operand.shape[0]:
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
def _update_struct(struct, numerator_condition):
    if numerator_condition:
        for i in range(struct.shape[0]-1, -1, -1):
            if struct[i,3] > (struct[i,1]-1)//2:
                temp = np.where((struct[i:,0]==struct[i,0]) & (struct[i:,1]==struct[i,1]))[0] + i
                struct[temp,3] = struct[i,3] - 1
                temp_1 = np.max(temp) + 1
                struct[temp_1:,3] = struct[temp_1:,1] - 1
                return True

        return False
    else:
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


@njit
def _get_profits_by_weight(weight, profit, index, num_test):
    overall_profit = np.zeros(num_test)
    num_test_1 = num_test - 1
    temp_profit = 1.0
    for i in range(index.shape[0]-2, -1, -1):
        temp = weight[index[i]:index[i+1]]
        max_ = np.where(temp==np.max(temp))[0] + index[i]
        if max_.shape[0] == 1:
            temp_profit *= profit[max_[0]]
        
        if i <= num_test_1:
            overall_profit[num_test_1-i] = temp_profit
    
    return overall_profit


@njit
def _get_geomean_profits_by_weights(weights, profit, index, num_test, target):
    two_d_geomean_profits = np.zeros((weights.shape[0], num_test))
    for i in range(weights.shape[0]):
        two_d_geomean_profits[i] = _get_profits_by_weight(weights[i], profit, index, num_test)
    
    test_start = index.shape[0] - num_test
    for i in range(num_test):
        two_d_geomean_profits[:,i] **= (1.0/(test_start+i))
    
    check_target = np.where(two_d_geomean_profits >= target, 1, 0)
    check_valid = np.full(weights.shape[0], 0)
    for i in range(weights.shape[0]):
        if (check_target[i]==1).any():
            check_valid[i] = 1
    
    temp = np.where(check_valid==1)[0]

    return temp, check_target[temp]


@njit
def _njit_fill_operand_many(formula, struct, idx, temp_0, temp_op, temp_1, target, current_5, operand, list_formula_0, list_formula_1, count, profit, index, num_test):
        start = -1
        if (formula[0:idx]==current_5[0:idx]).all():
            start = current_5[idx]
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

                valid_idx, check_target = _get_geomean_profits_by_weights(temp_0_new, profit, index, num_test, target)
                if valid_idx.shape[0] > 0:
                    temp_list_formula = np.full((valid_idx.shape[0], formula.shape[0]), 0) + formula
                    temp_list_formula[:,idx] = valid_operand[valid_idx]
                    x_1 = count[0]
                    x_2 = count[0] + valid_idx.shape[0]
                    list_formula_0[x_1:x_2] = temp_list_formula
                    list_formula_1[x_1:x_2] = check_target
                    count[0:3:2] += valid_idx.shape[0]
                
                current_5[:] = formula[:]
                current_5[idx] = operand.shape[0]
                if count[0] >= count[1] or count[2] >= count[3]:
                    return True
            else:
                temp_list_formula = np.full((valid_operand.shape[0], formula.shape[0]), 0) + formula
                temp_list_formula[:,idx] = valid_operand
                idx_new = idx + 2
                for i in range(valid_operand.shape[0]):
                    if _njit_fill_operand_many(temp_list_formula[i], struct, idx_new, temp_0_new[i], temp_op_new, temp_1_new[i], target, current_5, operand, list_formula_0, list_formula_1, count, profit, index, num_test):
                        return True

        return False


@njit
def _get_valid_op(formula, struct, idx, start):
    valid_op = np.full(2, 0)
    valid_op[start-2:] = 1

    if idx // 2 <= struct[0,1] // 2:
        valid_op[1] = 0
    
    return np.where(valid_op == 1)[0] + 2


# @njit
# def _get_threshold(weight, index, profit):
#     temp_profit = 1.0
#     thresholds = np.zeros(index.shape[0])
#     for i in range(index.shape[0]-2, -1, -1):
#         temp = weight[index[i]:index[i+1]]
#         max_temp = np.max(temp)
#         max_ = np.where(temp == max_temp)[0] + index[i]
#         if max_.shape[0] == 1:
#             temp_profit *= profit[max_[0]]
#             thresholds[index.shape[0]-1-i] = max_temp
#         else:
#             thresholds[index.shape[0]-1-i] = 10**200
    
#     thresholds[0] = np.min(thresholds[1:]) - 10**(-10)
#     return temp_profit**(1.0/(index.shape[0]-1)), thresholds


# @njit
# def _get_profit_by_weight_and_threshold(weight, profit, index, threshold):
#     temp_profit = 1.0
#     for i in range(index.shape[0]-2, -1, -1):
#         temp = weight[index[i]:index[i+1]]
#         max_value = np.max(temp)
#         if max_value <= threshold:
#             temp_profit *= 1.01
#         else:
#             max_ = np.where(temp == max_value)[0] + index[i]
#             if max_.shape[0] == 1:
#                 temp_profit *= profit[max_[0]]
#             else:
#                 temp_profit *= 1.0
    
#     return temp_profit**(1.0/(index.shape[0]-1))


# @njit
# def _find_max_threshold(formula, operand, index, profit):
#     weight = _calculate_formula(formula, operand)
#     geomean_profit, thresholds = _get_threshold(weight, index, profit)
#     max_threshold = thresholds[0]
#     max_profit = -1.0
#     for threshold in thresholds:
#         temp_profit = _get_profit_by_weight_and_threshold(weight, profit, index, threshold)
#         if temp_profit > max_profit:
#             max_profit = temp_profit
#             max_threshold = threshold
    
#     return geomean_profit, max_threshold, max_profit


# @njit
# def _get_value_invest_threshold(formula, threshold, test_operand, test_profit):
#     weight = _calculate_formula(formula, test_operand)
#     max_value = np.max(weight)
#     if max_value <= threshold:
#         return max_value, -1, 1.01
#     else:
#         max_ = np.where(weight == max_value)[0]
#         if max_.shape[0] == 1:
#             return max_value, max_[0], test_profit[max_[0]]
#         else:
#             return 0.0, -2, -1.0


@njit
def _get_threshold(weight, index, profit):
    temp_profit = 1.0
    thresholds = np.zeros(index.shape[0]-1, dtype=np.float64)
    profits_nguong = np.zeros(index.shape[0]-1)
    for i in range(index.shape[0]-2, -1, -1):
        temp = weight[index[i]:index[i+1]]
        max_temp = np.max(temp)
        max_ = np.where(temp == max_temp)[0] + index[i]
        if max_.shape[0] == 1:
            temp_profit *= profit[max_[0]]
            thresholds[index.shape[0]-2-i] = max_temp
            profits_nguong[index.shape[0]-2-i] = profit[max_[0]]
        else:
            thresholds[index.shape[0]-2-i] = 1e200
            profits_nguong[index.shape[0]-2-i] = 1.0
    
    min_ = np.min(thresholds)
    x_nguong = min_ - np.max(np.array([1e-9, 1e-9*min_]))
    return temp_profit**(1.0/(index.shape[0]-1)), thresholds, profits_nguong, x_nguong


@njit
def _find_max_threshold(formula, operand, index, profit):
    weight = _calculate_formula(formula, operand)
    geomean_profit, thresholds, profits_nguong, x_nguong = _get_threshold(weight, index, profit)
    
    max_profit = geomean_profit
    for x in thresholds:
        temp_profit = 1.0
        for i in range(0, thresholds.shape[0]):
            if thresholds[i] > x:
                temp_profit *= profits_nguong[i]
            else:
                temp_profit *= 1.01
        
        geo_ = temp_profit**(1.0/(index.shape[0]-1))
        if geo_ > max_profit:
            max_profit = geo_
            x_nguong = x
    
    return geomean_profit, x_nguong, max_profit


@njit
def _get_value_invest_threshold(formula, threshold, test_operand, test_profit):
    weight = _calculate_formula(formula, test_operand)
    max_value = np.max(weight)
    if max_value <= threshold:
        return max_value, -1, 1.01
    else:
        max_ = np.where(weight == max_value)[0]
        if max_.shape[0] == 1:
            return max_value, max_[0], test_profit[max_[0]]
        else:
            return 0.0, -2, -1.0