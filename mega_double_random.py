import pandas as pd
import numpy as np
import os


def generate_mega_numbers():

    file_path = os.path.join(
        os.getcwd(),
        "lottnums",
        "Lottery_Mega_Millions_true.csv"
    )

    dfis = pd.read_csv(file_path)

    mins = {
        1: dfis['Line1'].min(),
        2: dfis['Line2'].min(),
        3: dfis['Line3'].min(),
        4: dfis['Line4'].min(),
        5: dfis['Line5'].min(),
        6: dfis['Linemain'].min()
    }

    maxs = {
        1: dfis['Line1'].max(),
        2: dfis['Line2'].max(),
        3: dfis['Line3'].max(),
        4: dfis['Line4'].max(),
        5: dfis['Line5'].max(),
        6: dfis['Linemain'].max()
    }

    valueAll = [0]

    for i in range(1, 6):

        previous_value = valueAll[-1]

        value1 = np.random.randint(mins[i], maxs[i] + 1)
        value2 = dfis[f'Line{i}'].sample(n=1).iloc[0]

        while (value1 != value2) or (value2 <= previous_value):
            value1 = np.random.randint(mins[i], maxs[i] + 1)
            value2 = dfis[f'Line{i}'].sample(n=1).iloc[0]

        valueAll.append(value2)

    value1Main = np.random.randint(mins[6], maxs[6] + 1)
    value2Main = dfis['Linemain'].sample(n=1).iloc[0]

    while value1Main != value2Main:
        value1Main = np.random.randint(mins[6], maxs[6] + 1)
        value2Main = dfis['Linemain'].sample(n=1).iloc[0]

    valueAll.append(value2Main)

    string_array = [str(element) for element in valueAll[1:]]

    return string_array
