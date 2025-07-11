import pickle
import matplotlib.pyplot as plt

dtn = "2025-07-06_23:24:17"
with open(f"controller/logs/logs_l_task/{dtn}/plot3d_pkl.pkl", "rb") as f:
    fig = pickle.load(f)

fig.show()
input()