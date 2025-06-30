import mujoco
import mujoco.viewer
import time
import numpy as np
from utils import *
np.set_printoptions(
    linewidth=200,     # Wider output (default is 75)
    threshold=np.inf,  # Print entire array, no summarization with ...
    precision=8,       # Show more decimal places
    suppress=False     # Do not suppress small floating point numbers
)


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

    # model_path = "assets/ur3e/ur3e.xml"
    # model_path = "assets/ur3e_raw.xml"
    # model_path = "assets/2f85/2f85.xml"
    # model_path = "assets/main.xml"
    # model_path = "assets/mug/mug.xml"
    # model_path = "assets/ur3e_fish.xml"
    model_path = "assets/ur3e_2f85.xml"
    
    

    m, d = load_model(model_path)
    
    # mug   = 7  nq, 6  nv, 0 nu
    # ur3e  = 6  nq, 6  nv, 6 nu
    # 2f85  = 8  nq, 8  nv, 1 nu
    # total = 21 nq, 20 nv, 7 nu


    # reset(m, d)

    viewer = mujoco.viewer.launch_passive(m, d)
    # python -m mujoco.viewer --mjcf=./model/ur3e.xml

    start = time.time()
    t = time.time() - start

    while t < 1000:
        mujoco.mj_step(m, d)
        viewer.sync()

        # print("nq:", m.nq, " nv:", m.nv, " nu:", m.nu)
        print(d.site("right_pad1_site").xpos.copy())
        
        # print("nq: ", m.nq)
        # print("nv: ", m.nv)
        # print("nu: ", m.nu)
        # print("________________")

        # print(get_grasp_force(d))
        # print(get_grasp_bool(d))


        # time.sleep(0.01)
        t = time.time() -  start




if __name__ == "__main__":
    main()