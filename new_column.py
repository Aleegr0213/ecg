import pandas as pd

normal_signals_dt = pd.read_csv('ptbdb_normal.csv')
abnormal_signals_dt = pd.read_csv('ptbdb_abnormal.csv')
path_new_normal = pd.read_csv("new_normal_signal.csv")
normal_signals_dt.insert(0, "0", 0)
normal_signals_dt.to_csv('new_normal_signal.csv', index=False)

print(path_new_normal)


