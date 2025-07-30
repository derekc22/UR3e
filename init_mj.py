import mujoco
import mujoco.viewer
import time
import numpy as np
from controller.controller_utils import *
from utils.gym_utils import *
np.set_printoptions(
    linewidth=400,     # Wider output (default is 75)
    threshold=np.inf,  # Print entire array, no summarization with ...
    precision=8,       # Show more decimal places
    suppress=False     # Do not suppress small floating point numbers
)
import os







def main():

    # m_path = "assets/ur3e/ur3e.xml"
    # m_path = "assets/ur3e_raw.xml"
    # m_path = "assets/2f85/2f85.xml"
    m_path = "assets/main.xml"
    # m_path = "assets/mug/mug.xml"
    # m_path = "assets/ur3e_fish.xml"
    # m_path = "assets/ur3e_2f85.xml"
    
    

    m, d = load_model(m_path)
    
    # mug   = 7  nq, 6  nv, 0 nu
    # ur3e  = 6  nq, 6  nv, 6 nu
    # 2f85  = 8  nq, 8  nv, 1 nu
    # total = 21 nq, 20 nv, 7 nu


    reset(m, d, "down")


    viewer = mujoco.viewer.launch_passive(m, d)
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
    # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY


    # python -m mujoco.viewer --mjcf=./m/ur3e.xml

    start = time.time()
    t = time.time() - start


    collision_cache = init_collision_cache(m)
    # for  c in collision_cache:
    #     for i in c:
    #         print(get_body_name(m, i))
    # exit()
    
    while t < 1000:
        mujoco.mj_step(m, d)
        viewer.sync()
        t = time.time() -  start
        
        # print(get_site_velp(m, d, "tcp"))

        # print(get_body_size(m, "fish"))
        # get_mug_toppled(m, d)
        # print(get_mug_toppled(m, d))

        # print(get_self_collision(m, d, collision_cache))
        # print(get_table_collision(m, d, collision_cache))
        # print(get_boolean_grasp_contact(d))
        # print(get_grasp_contact(d))
        # print(get_block_grasp_state(m, d))

        # get_table_collision(m, d, collision_cache)
        
        # print(get_children_deep(m, id_))
        # print(get_children_deep(m, get_body_id(m, "robot_base")))
        # print(id_)
        # x = [get_body_id(m, f) for f in finger_bodies ]
        # print(x)


        # print("nq: ", m.nq)
        # print("nv: ", m.nv)
        # print("nu: ", m.nu)
        # print("________________")
        

        # body_id = m.body(name="shoulder_link").id
        # body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "wrist_3_sensor")
        # mujoco.mj_forward(m, d)
        # position = d.xpos[body_id]
        # print(position)
        # print(get_2f85_home2(m, d))
        # print(get_2f85_home(m, d))
        # print(get_2f85_xpos(m, d))
        # exit()
        
        # print(m.jnt_range)
        # print(len(m.jnt_range))
        
        # print(m.qpos0)
        # print(m.qvel0)
        # print(get_body_xpos(m, d, "fish"))
        # exit()
        



if __name__ == "__main__":
    main()