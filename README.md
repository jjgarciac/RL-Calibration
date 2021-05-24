# RL-Calibration
Implementation of continious control experiments in "Calibrated Model-Based Deep Reinforcement Learning" (Malik et. al, 2019). 

To run PETS model:
python -env cartpole -k 1 -calibrate 

To run PETS++:
python -env cartpole -k 3

# Requirements
tensorflow <2.4.0>
tensorflow_probability <0.12.1>
gym <0.18.0>
tf_agents <0.7.1>
mujoco_py <2.0.2.9>
numpy <1.19.2>
sklearn <0.23.0>
json <2.0.9>
argparse <1.1>
