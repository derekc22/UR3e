import pickle
import argparse

parser = argparse.ArgumentParser(description="Specify trajectory pkl")
parser.add_argument("-f", "--file", type=str, help="Specify trajectory pkl")
args = parser.parse_args()

with open(f"controller/logs/logs_l_task/{args.file}/plot3d_pkl.pkl", "rb") as f:
    fig = pickle.load(f)

fig.show()
input()