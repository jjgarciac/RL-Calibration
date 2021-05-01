import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from utils import empirical_cdf
from tf_agents.agents import tf_agent
from sklearn.model_selection import train_test_split

tfd = tfp.distributions
tfk = tf.keras
tfkl = tf.keras.layers
U = tfd.Uniform()

class Agent(tf_agent.TFAgent):
    def __init__(self, policy, train_step_counter=None):
        super(Agent, self).__init__(time_step_spec=policy.time_step_spec,
                                        action_spec=policy.action_spec,
                                        policy=policy,
                                        collect_policy=policy,
                                        train_sequence_length=None,
                                        train_step_counter=train_step_counter)

    def _train(self, experience, weights=None):
        self.fit(experience.action, experience.observation)
        self.train_step_counter.assign_add(1)
        return tf_agent.LossInfo((), ())

    def fit(self, a, s):
        if len(a.shape)<len(s.shape):
            a = a[...,None]
        # Necessary cast if a is not same dtype as s
        a = tf.cast(a, s.dtype)
        if self.policy.obs_preproc is not None:
            s = self.policy.obs_preproc(s)

        x = tf.concat([s[:, :-1, :], a[:, :-1, :]], axis=-1)
        x = tf.reshape(x, [-1, x.shape[-1]])
        y = tf.reshape(s[:, 1:, :] - s[:, :-1, :], [-1, s.shape[-1]])

        train_x, cal_x, train_y, cal_y = train_test_split(x.numpy(), y.numpy(), test_size=0.2)
        # Select a subset of random indices for each num_net
        idx = np.random.randint(train_x.shape[0], size=[self.policy.num_nets, train_x.shape[0]]) 
        #idx = tf.random.uniform([self.policy.num_nets, train_x.shape[0]], 0, 
        #        train_x.shape[0], tf.int32)
        #train_x = tf.gather(train_x, idx)
        #train_y = tf.gather(train_y, idx)
        
        self.policy.model.fit(train_x[idx], train_y[idx])

        # TODO: Add calibration split
        if self.policy.calibrator is not None:
            idx = np.random.randint(cal_x.shape[0], 
                    size=[self.policy.num_nets, cal_x.shape[0]]) 
            # print(f"idx: {idx.shape}")
            cdf_pred = self.policy.model(cal_x[idx]).cdf(cal_y[idx])
            # print(f"pred_dist: {self.policy.model(cal_x[idx])}")
            # print(f"cal_x: {cal_x.shape}")
            # print(f"cal_x[idx]: {cal_x[idx].shape}")
            # print(f"cal_y[idx]: {cal_y[idx].shape}")
            # print(f"cal_y: {cal_y.shape}")
            cdf_pred = tf.reshape(cdf_pred, [-1, 1])
            cdf_true = empirical_cdf(cdf_pred)
            # print(f"cdf_true: {cdf_true.shape}")
            # print(f"cdf_pred: {cdf_pred.shape}")
            self.policy.calibrator.fit(cdf_pred, cdf_true)

