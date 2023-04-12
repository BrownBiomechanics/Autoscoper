NUM_CAMERAS = 2
# [Example 1 - Start]
from PyAutoscoper.connect import (
    AutoscoperConnection,
    OptimizationMethod,
    CostFunction,
    OptimizationInitializationHeuristic,
)

autoscoperSocket = AutoscoperConnection()
autoscoperSocket.is_connected()
# [Example 1 - End]

# [Example 2 - Start]
trial_config = "path/to/trial_config.cfg"
autoscoperSocket.loadTrial(trial_config)
filter_settings = "path/to/filter_settings.vie"
for camera_i in range(NUM_CAMERAS):
    autoscoperSocket.loadFilters(camera_i, filter_settings)
# [Example 2 - End]

# [Example 3 - Start]
tracking_data = "path/to/tracking_data.tra"
tracking_data_out = "path/to/tracking_data_out.tra"
autoscoperSocket.loadTrackingData(0, tracking_data)
autoscoperSocket.saveTracking(0, tracking_data_out)
# [Example 3 - End]

# [Example 4 - Start]
import random as rand

for frame in range(10):
    autoscoperSocket.setFrame(frame)
    current_pose = autoscoperSocket.getPose(0, frame)
    # Add a random number between -1 and 1 to each pose value
    new_pose = [current_pose[i] + rand.uniform(-1, 1) for i in range(6)]
    autoscoperSocket.setPose(0, frame, new_pose)
# [Example 4 - End]

# [Example 5 - Start]
autoscoperSocket.optimizeFrame(
    volume=0,
    frame=0,
    repeats=1,
    max_itr=100,
    min_lim=-1.0,
    max_lim=1.0,
    max_stall_itr=10,
    dframe=1,
    opt_method=OptimizationMethod.PARTICLE_SWARM_OPTIMIZATION,
    cf_model=CostFunction.NORMALIZED_CROSS_CORRELATION,
    opt_init_heuristic=OptimizationInitializationHeuristic.PREVIOUS_FRAME,
)
# [Example 5 - End]

# [Example 5.1 - Start]
autoscoperSocket.trackingDialog(volume=0, start_frame=0, end_frame=10)
# [Example 5.1 - End]

# [Example 6 - Start]
import random as rand
from PyAutoscoper.connect import (
    AutoscoperConnection,
    OptimizationMethod,
    CostFunction,
    OptimizationInitializationHeuristic,
)

# Create a socket connection to Autoscoper
autoscoperSocket = AutoscoperConnection()

# Load a trial
autoscoperSocket.loadTrial("path/to/trial.cfg")
# Load filters
autoscoperSocket.loadFilters(0, "path/to/filters.vie")
autoscoperSocket.loadFilters(1, "path/to/filters.vie")
# Load initial tracking data
for volume in range(3):
    autoscoperSocket.loadTrackingData(volume, f"path/to/tracking_data_volume_{volume}.tra")

NUM_FRAMES = 100
frame_skip = 1
for volume in range(3):
    for frame in range(0, NUM_FRAMES, frame_skip):
        autoscoperSocket.setFrame(frame)
        current_pose = autoscoperSocket.getPose(volume, frame)
        # Add a random number between -1 and 1 to each pose value
        new_pose = [current_pose[i] + rand.uniform(-1, 1) for i in range(6)]
        autoscoperSocket.setPose(volume, frame, new_pose)

        # Optimize tracking data
        autoscoperSocket.optimizeFrame(
            volume=volume,
            frame=frame,
            repeats=1,
            max_itr=100,
            min_lim=-1.0,
            max_lim=1.0,
            max_stall_itr=10,
            dframe=1,
            opt_method=OptimizationMethod.PARTICLE_SWARM_OPTIMIZATION,
            cf_model=CostFunction.NORMALIZED_CROSS_CORRELATION,
            opt_init_heuristic=OptimizationInitializationHeuristic.CURRENT_FRAME,
        )

    autoscoperSocket.saveTracking(volume, f"path/to/tracking_data_volume_{volume}_out.tra")

autoscoperSocket.closeConnection()
# [Example 6 - End]

# [Example 7 - Start]
import subprocess as sp
import signal, os

executable = "path/to/Autoscoper.exe"

# Launch Autoscoper
AutoscoperProcess = sp.Popen([executable])

# check if Autoscoper is running
if AutoscoperProcess.poll() is None:
    print("Autoscoper is running")
else:
    print("Autoscoper is not running")

# Kill Autoscoper
os.kill(AutoscoperProcess.pid, signal.SIGTERM)

# check if Autoscoper is running
if AutoscoperProcess.poll() is None:
    print("Autoscoper is running")
else:
    print("Autoscoper is not running")
# [Example 7 - End]
