from sqlite3 import Connection
import numpy as np
from Methods.base import decode_formula, convert_arrF_to_strF
import pandas as pd
import json


def get_list_table():
    return '''SELECT name FROM sqlite_master WHERE type = "table";'''


def create_table(len_formula, list_field, cycle):
    list_formula_col = [f'"E{i}" INTEGER NOT NULL,' for i in range(len_formula)]
    list_field_col = [f'"{field[0]}" {field[1]},' for field in list_field]
    temp = "\n    "
    return f'''CREATE TABLE "{cycle}_{len_formula}" (
    "id" INTEGER NOT NULL,
    {temp.join(list_formula_col)}
    {temp.join(list_field_col)}
    PRIMARY KEY ("id")
)'''


def insert_rows(table_name, list_of_list_value):
    if len(list_of_list_value) == 0:
        return ";"

    temp_list = []
    for list_value in list_of_list_value:
        temp = ""
        for value in list_value:
            if type(value) == str:
                temp += f'"{value}",'
            else:
                temp += f"{value},"

        temp_list.append(f"({temp[:-1]}),")

    text = "\n".join(temp_list)

    return f'''INSERT INTO "{table_name}" VALUES\n{text[:-1]};'''


def _top_n_by_column(table_name, column, n_row, select_all):
    num_operand = int(table_name.split("_")[1])
    text_1 = '"id"'
    for i in range(num_operand):
        text_1 += f', "E{i}"'

    if select_all:
        text_1 = "*"
    else:
        text_1 += f', "{column}"'

    return f'SELECT {text_1} FROM "{table_name}" ORDER BY "{column}" DESC LIMIT {n_row};'


def top_n_by_column(time, column, n_row, db_file_path, num_data_operand=None, select_all=False):
    connection = Connection(db_file_path)
    cursor = connection.cursor()
    cursor.execute(get_list_table())
    list_table = [t_[0] for t_ in cursor.fetchall() if t_[0].startswith(str(time))]
    list_of_list_value = []
    opr = db_file_path.replace(db_file_path[db_file_path.index("METHOD"):], "operand_names.json")
    with open(opr, "r") as f:
        operands = json.load(f)

    if num_data_operand is None:
        num_data_operand = len(operands.keys())

    for table in list_table:
        query = _top_n_by_column(table, column, n_row, select_all)
        print(query)
        cursor.execute(query)
        list_value = cursor.fetchall()
        n_op = int(table.split("_")[1])
        for i in range(len(list_value)):
            temp = np.array(list(list_value[i][1:n_op+1]))
            ct = decode_formula(temp, num_data_operand).astype(int)
            list_value[i] = [list_value[i][0]] + [convert_arrF_to_strF(ct)] + list(list_value[i][n_op+1:])
        list_of_list_value += list_value

    if select_all:
        list_col = ["id", "CT"]
        cursor.execute(f"PRAGMA table_info('{table}');")
        for r_ in cursor.fetchall()[n_op+1:]:
            list_col.append(r_[1])
    else:
        list_col = ["id", "CT", column]

    data = pd.DataFrame(list_of_list_value, columns=list_col)
    data.sort_values(column, inplace=True, ignore_index=True, ascending=False)
    connection.close()
    return data.loc[:n_row-1]
