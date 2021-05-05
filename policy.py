from typing import Optional, Text, Sequence
from typing import cast
import tensorflow as tf
import tensorflow_probability as tfp
from utils import *
from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.typing import types
from tf_agents.specs import tensor_spec
from tf_agents.utils import nest_utils

U = tfp.distributions.Uniform()

class Policy(tf_policy.TFPolicy):
    """ Policy
    """
    def __init__(self, time_step_spec, action_spec, policy_state_spec=(), 
                model='p', num_nets=1, k=1, plan_hor=12, npart=5, pop_size=10,
                ts_sampler='ts1', a_sampler='random', env='cartpole', calibrate=None):
        super(Policy, self).__init__(time_step_spec, action_spec, policy_state_spec=())
        self.num_nets = num_nets
        self.do = time_step_spec.observation.shape[0]
        try:
            self.du = action_spec.shape[0]
        except IndexError:
            self.du = 1
        self.obs_cost, self.act_cost, self.obs_preproc = get_env_cost(env)
        if self.obs_preproc is not None:
           self.do_preproc =  self.obs_preproc(np.ones([1, self.do])).shape[-1]
        self.k = k
        self.model = get_model(model, self.num_nets, self.du, self.do, self.do_preproc, k)
        self.a_sampler = get_a_sampler(a_sampler, action_spec)
        self.ts_sampler = get_ts_sampler(ts_sampler)
        self.plan_hor = plan_hor
        self.pop_size = pop_size
        self.npart = npart
        self.calibrator, self.inv_cal = None, None
        if calibrate:
            self.calibrator, self.inv_cal = get_calibrator()
        self.act_max = tf.cast(action_spec.maximum, tf.float32)
        self.act_min = tf.cast(action_spec.minimum, tf.float32)

    def _action(self, time_step: ts.TimeStep, policy_state: types.NestedTensor = (),
        seed: Optional[types.Seed] = None) -> policy_step.PolicyStep:
        action_spec = cast(tensor_spec.BoundedTensorSpec, self.action_spec)
        act_seq = self.a_sampler.sample([self.plan_hor, self.pop_size, self.npart, self.du])
        # Cast to work with DNN
        act_seq = tf.cast(act_seq, tf.float32)
        obs = tf.cast(time_step.observation[None, ...], tf.float32)
        act_seq = tf.clip_by_value(act_seq, self.act_min, self.act_max)

        obs = tf.tile(obs, [self.pop_size, self.npart, 1]) 
        cost = tf.zeros([self.pop_size, self.npart], tf.float32)
        for act in act_seq:
            cost += self.act_cost(act) + self.obs_cost(obs)
            pre_obs = self.obs_preproc(obs)
            obs = self.ts_sampler(self.model, pre_obs, obs, act, self.pop_size, self.npart, 
                    self.num_nets, self.k, self.calibrator, self.inv_cal)
        
        # Assign high cost to nan values
        cost = tf.where(tf.math.is_nan(cost), 1e6 * tf.ones_like(cost), cost)
        idx = tf.argmin(tf.reduce_mean(cost, axis=1))
        act = act_seq[0, idx, 0, :]

        outer_dims = nest_utils.get_outer_shape(time_step, self._time_step_spec)
        
        # Cast to take step
        action_ = tf.cast(act[None, ...], action_spec.dtype)
        policy_info = tensor_spec.sample_spec_nest(self._info_spec, outer_dims=outer_dims)
        step = policy_step.PolicyStep(action_, policy_state, policy_info)
        return step 

