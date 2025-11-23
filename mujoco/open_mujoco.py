import mujoco
import mujoco.viewer
import time

# Load an XML model (use one of MuJoCo’s included models or your own)
model = mujoco.MjModel.from_xml_path("./scene.xml")
data = mujoco.MjData(model)

# Launch the viewer (interactive window)
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Run the simulation loop
    while viewer.is_running():
        step_start = time.time()

        mujoco.mj_step(model, data)

        # Optionally sync viewer every step
        viewer.sync()

        # Control loop timing (~60 Hz)
        time_until_next_step = 1/60 - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
