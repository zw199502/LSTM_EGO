# LSTM_EGO
Learn to navigate in dynamic environments with normalized LiDAR scans

# Configuration
- Pytorch
- Cython

# Compile LiDAR scan library
In C_library, ```python setup.py build_ext --inplace```

# Choose environment
In main.py, ```parser.add_argument("--env", default="crowd_real_all_circles")```. "crowd_sim" stands for basic environments only having 5 dynamic circle humans; "crowd_sim_complex" represent complex scenarios mixing dynamic circle humans (maximum number is 4) and static rectangle obstacles (maximum number is 3); "crowd_real" denotes real-world environments where the robot moves on a narrow plane; "crowd_real_all_circles" are the environments for sim-to-real transfer.
