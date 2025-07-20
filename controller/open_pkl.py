import pickle
import argparse

parser = argparse.ArgumentParser(description="Specify trajectory pkl")
parser.add_argument("-f", "--folder", type=str, help="Specify trajectory pkl folder")
args = parser.parse_args()

with open(f"{args.folder}/plot3d_pkl.pkl", "rb") as f:
    fig = pickle.load(f)

fig.show()
input()