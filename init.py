import mujoco
import mujoco.viewer
import time





def load_model(model_path):
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)

    return m, d



def main():

    model_path = "model/ur3e.xml"

    m, d = load_model(model_path)

    viewer = mujoco.viewer.launch_passive(m, d)
    # python -m mujoco.viewer --mjcf=./model/ur3e.xml

    while True:
        mujoco.mj_step(m, d)
        viewer.sync()
        time.sleep(0.01)




if __name__ == "__main__":
    main()