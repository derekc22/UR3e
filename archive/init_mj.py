import mujoco
import mujoco.viewer
import time
import numpy as np



def load_model(model_path):
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)

    return m, d


def reset(m, d, intialization='home'):
    init_qp = np.array(m.keyframe(intialization).qpos)
    mujoco.mj_resetData(m, d) 
    d.qpos[:] = init_qp
    mujoco.mj_step(m, d)




def main():

    model_path = "archive/model/ur3e.xml"

    m, d = load_model(model_path)

    reset(m, d)

    viewer = mujoco.viewer.launch_passive(m, d)
    # python -m mujoco.viewer --mjcf=./model/ur3e.xml

    start = time.time()
    t = time.time() - start

    while t < 1000:
        mujoco.mj_step(m, d)
        viewer.sync()

        print("dimensions: ", m.nv)

        # time.sleep(0.01)
        t = time.time() -  start




if __name__ == "__main__":
    main()