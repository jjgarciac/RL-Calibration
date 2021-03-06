import gym
import keras
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import imageio
import IPython
import base64
import json
from argparse import ArgumentParser
from tf_agents.environments import batched_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import suite_gym
from tf_agents.environments import suite_mujoco
from pynverse import inversefunc

tfk = tf.keras
tfd = tfp.distributions
U = tfd.Uniform()

class FC(keras.layers.Layer):
    """Densely connected NN layer op.
    Args:
        units: Num of units to output.
        num_nets: Num of independent ANNs
        feat_size: Dimensionality of state and action space (i.e. du+do).
        activation: (Optional) 1-argument callable. Activation function to apply to
          outputs.
    Returns:
        tf.Tensor. Output of dense connection.
    """
    def __init__(self, units, num_nets, feat_size, activation=tf.nn.sigmoid, kernel_regularizer=None):
        super(FC, self).__init__()
        self.units = units
        self.num_nets = num_nets
        self.feat_size = feat_size
        self.activation = activation
        self.kernel_regularizer = kernel_regularizer

        self.w = self.add_weight(
            shape=(self.num_nets, self.feat_size, self.units),
            initializer="random_normal",
            trainable=True,
            regularizer=self.kernel_regularizer,
            name="w"
        )
        self.b = self.add_weight(
            shape=(self.num_nets, 1, self.units), 
            initializer="random_normal", 
            trainable=True,
            name="b"
        )

    def call(self, inputs):
        outputs = tf.matmul(inputs, self.w) + self.b
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

def embed_mp4(filename):
    """Embeds an mp4 file in the notebook."""
    video = open(filename,'rb').read()
    b64 = base64.b64encode(video)
    tag = '''
    <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
    Your browser does not support the video tag.
    </video>'''.format(b64.decode())

    return IPython.display.HTML(tag)

def video_agent(env, video_env, policy, video_filename, frames=500):
    """Saves a video of the agent in the environment.

    Args:
        env: TF Environement.
        video_env: Py Environement.
    """
    time_step = env.reset()
    with imageio.get_writer(video_filename, fps=60) as video:
        video.append_data(video_env.render())
        for _ in range(frames):
            video.append_data(video_env.render())
            time_step = env.step(policy.action(time_step))
    embed_mp4(video_filename)

def load_args(argspath):
    """ Load arguments from argspath file.
    """
    parser = ArgumentParser()
    args, unknown = parser.parse_known_args()
    with open(argspath, 'r') as f:
        args.__dict__ = json.load(f)
    return args 

def get_env(env='cartpole'):
    """ TF Environment switch.

    Args:
        env: Environement name.
    Returns:
        TF Envitonment.
        Py Environment.
    """
    if env == 'cartpole':
        env = suite_gym.load('CartPole-v0')
    if env=='cartpolem':
        env = suite_gym.load('InvertedPendulum-v2')
    if env=='pusher':
        env = suite_mujoco.load('Pusher-v2')
    if env=='cheetah':
        env = suite_gym.load('HalfCheetah-v2')
    if env=='reacher':
        env = suite_gym.load('Reacher-v2')
    return tf_py_environment.TFPyEnvironment(env), env

def get_model(model, num_nets, du, do, do_preproc, k, hidden_size=100):
    """Creates and compiles an ensemble of Gaussian-Mixtures, each parametrized by 
    a Neural-Net. PETS corresponds to (model="P", num_nets=5, k=1). PETS++ 
    corresponds to (model="P", num_nets=5, k=3).
    Args:
        model: Type of model.
        num_nets: Number of networks in the ensemble.
        du: Dimensionality of action space.
        do: Dimensionality of state space.
        do_preproc: Dimensionality of pre-processed state space.
        k: Number of Mixture Components.
        hidden_size: Size of hidden layer for each network in ensemble.
    Returns:
        tf.Model. Transition probability model p(s|a,s).
    """
    if model[0]=="P":
        convert_to_tensor_fn = lambda s: s.sample()
    else: 
        convert_to_tensor_fn = lambda s: s.mean()

    GMM = lambda t: tfd.MixtureSameFamily( 
                components_distribution = tfd.Independent(tfd.Normal(
                        loc=tf.transpose(tf.stack(
                            [t[..., do*i:do*(1+i)] for i in range(k)]), [1,2,0,3]), 
                        scale=tf.transpose(tf.stack(
                            [1e-5 + tf.exp(t[..., do*i:do*(1+i)]) for i in range(k, 2*k)]), [1,2,0,3])), 
                    reinterpreted_batch_ndims=1), 
                mixture_distribution =tfd.Categorical(probs=tf.nn.softmax(t[..., do*2*k:])))

    make_distribution_fn = GMM
    fc_output = do*2*k + k 

    ps_as = tfk.models.Sequential([
            FC(hidden_size, num_nets, du+do_preproc, tf.keras.activations.swish, 
                tf.keras.regularizers.L2(0.000025)),
            FC(hidden_size, num_nets, hidden_size, tf.keras.activations.swish, 
                tf.keras.regularizers.L2(0.00005)),
            FC(fc_output, num_nets, hidden_size, None, 
                tf.keras.regularizers.L2(0.00075)),
            tfp.layers.DistributionLambda(
                make_distribution_fn=make_distribution_fn,
                convert_to_tensor_fn=convert_to_tensor_fn)])
    
    negloglik = lambda y, rv_y: -rv_y.log_prob(y)
    ps_as.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss=negloglik)
    return ps_as

def get_calibrator():
    """Calibration function.
    Args:
        None
    Returns:
        tf.Model. Output of dense connection.
        function. Inverse call to calibration.
    """
    r = tfk.models.Sequential([
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    cal_loss = lambda y, y_pred: tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(y, y_pred))
    r.compile(optimizer=tf.optimizers.Adam(learning_rate=5e-2), loss=cal_loss)
    
    def inv_cal(r, y):
        try:
            w = r.layers[0].kernel
            b = r.layers[0].bias
        except AttributeError:
            w = 1.
            b = 0.
        out  = tf.math.log(y/(1 - y))
        return (out - b)/w

    return r, inv_cal

def random_sampler(action_spec):
    """Returns action-sequence sampler. 
    Args:
        action_spec: BoundedArraySpec for the actions in the environment.
    Returns:
        Distribution to sample action-sequence.
    """
    mx = action_spec.maximum 
    mn = action_spec.minimum
    if action_spec.dtype == tf.int64:
        return tfd.Categorical(probs=[1/(mx+1)]*(mx+1))
    else:
        return tfd.Uniform(mn, mx)

def mm_quantile(mm, k, p, n_samples):
    """Computes quantile of Mixture Model from samples.
    Args:
        mm: Mixture Model.
        k: Number of components in Mixture Model.
        p: Probability to define quantile Pr(X<p).
        n_samples: Number of samples to use.
    """
    samples = mm.sample(k*n_samples)
    probs = mm.cdf(samples)
    delta = tf.math.abs(probs - p[None, ...])
    idx = tf.argmin(tf.reduce_sum(delta, axis=[-2, -1]))
    return samples[idx] 

def prepare_input_for_model(x, num_nets, npart):
    """ Divides particles amongst networks in ensemble.
    Args:
        x: tf.Tensor 
        num_nets: Number of networks in ensemble.
        npart: Number of particles.
    Returns:
        tf.Tensor 
    """
    dim = x.shape[-1]
    tmp = tf.reshape(x, [-1, num_nets, npart // num_nets, dim])
    tmp = tf.transpose(tmp, [1,0,2,3])
    return tf.reshape(tmp, [num_nets, -1, dim]) 

def flatten_delta(delta_obs, num_nets, pop_size, npart):
    """ Reshapes model output to match sampled trajectories shape. 
    """
    tmp = tf.reshape(delta_obs, [num_nets, pop_size, npart//num_nets, delta_obs.shape[-1]])
    tmp = tf.transpose(tmp, [1,0,2,3]) #[pop_size, num_nets, npart//num_nets, dim]
    return tf.reshape(tmp, [-1, delta_obs.shape[-1]]) #[pop_size*npart, dim] if npart is multiple of num_nets

def ts1(model, pre_obs, obs, act, pop_size, npart, num_nets, num_comp, calibrator=None, inv_cal=None):
    """Trajectory sampling scheme proposed in github.com/kchua/handful-of-trials/blob/master/dmbrl/controllers/MPC.py#L318
    Args:
        model: Transition Probability
        pre_obs: Pre-processed Observation
        obs: Observation
        act: Action
        pop_size: Number of sequences.
        npart: Number of particles per sequence.
        num_nets: Number of Mixture Models in ensemble.
        num_comp: Number of components in each Mixture Model.
        calibrator: Calibration function.
        inv_cal: Inverse-Calibration function.
    Returns:
        tf.Tensor. Next observation.
    """
    
    sort_idx = tf.nn.top_k(U.sample([pop_size, npart]), k=npart).indices #[pop_size, npart, npart]
    
    tmp = tf.range(pop_size)[:, None] #[pop_size, 1]
    tmp = tf.tile(tmp, [1, npart]) #[pop_size, npart] Array of [[0,...,0], ..., [pop_size-1,...,pop_size-1]]
    tmp = tmp[:,  :, None] #[pop_size, npart, 1]
    
    idx = tf.concat([tmp, sort_idx[:,:,None]], axis=-1) #[pop_size,npart,2]
    tmp_obs = tf.gather_nd(pre_obs, idx) #[pop_size, npart, do']

    s = prepare_input_for_model(tmp_obs, num_nets, npart)
    a = prepare_input_for_model(act, num_nets, npart)

    inputs = tf.concat([s, a], axis=-1)
    pred_dist = model(inputs)

    if calibrator is not None:
        p = U.sample([num_nets, pop_size*(npart//num_nets)])
        ps = inv_cal(calibrator, p)
        ps = tf.clip_by_value(ps, 1e-6, 1 - 1e-6)
        if num_comp == 1: #Check if only one Gauss component => Quantile exists
            delta_obs = pred_dist.components_distribution.distribution.quantile(ps[..., None, None])
        else:
            delta_obs = mm_quantile(pred_dist, num_comp, ps, n_samples=10)
    else:
        delta_obs = pred_dist.sample()

    # Reshape delta_obs to match obs shape.
    delta_obs = flatten_delta(delta_obs, num_nets, pop_size, npart)
    delta_obs = tf.reshape(delta_obs, [pop_size, npart, obs.shape[-1]])

    # Another permutation of the particles while keeping them on the same population.
    # TODO: Verify if model converges faster without this. This makes trajectories inconsistent.
    sort_idx = tf.nn.top_k(-sort_idx,k=npart).indices 
    idx = tf.concat([tmp, sort_idx[:, :, None]], axis=-1)
    delta_obs = tf.gather_nd(delta_obs, idx)

    return delta_obs + obs

def get_a_sampler(a_sampler, action_spec):
    """Action sampler switch.
    Args:
        a_sampler: Action sampler identifier.
        action_spec: Environment action's BoundedArraySpec.
    Returns:
        tfp.distribution. Action sampler.
    """
    if a_sampler == 'random':
        return random_sampler(action_spec)
    if a_sampler == 'CEM':
        return None

def get_ts_sampler(ts_sampler):
    """Trajectory Sampler switch.
    Args:
        ts_sampler: Trajectory sampler identifier.
    Returns:
        function. trajectory sampler.
    """
    if ts_sampler == 'ts1':
        return ts1

def empirical_cdf(cdf_pred):
    """Computes empirical cdf from cdf predictions.
    Args:
        cdf_pred: tf.Tensor. 
    """
    return tf.concat([tf.reduce_mean(tf.cast(cdf_pred<cdf_pred[i], tf.float32)) for i in range(cdf_pred.shape[0])], axis=0)

def get_env_cost(env):
    """Oracle cost functions and preprocessing functions. 
    Args:
        env: Environment name.
    Returns:
        obs_cost: Observation cost function.
        act_cost: Action cost function.
        obs_preproc: Observation pre-processing function.
    """
    if env=='cartpole':
        def obs_cost(x):
            x0, theta = x[..., :1], x[..., 1:2]
            h = tf.concat([x0 - 0.6 * tf.sin(theta), -0.6 * tf.cos(theta)], axis=-1)
            return tf.cast(-tf.exp(-tf.reduce_sum(tf.square(h - np.array([[0.0, 0.6]])), 
                            axis=-1) / (0.6 ** 2)), tf.float32)

        def act_cost(x):
            return 0.01 * tf.reduce_sum(tf.square(x), axis=-1)

        def obs_preproc(x):
            return tf.concat([tf.sin(x[..., 1:2]), tf.cos(x[..., 1:2]), 
                            x[..., :1], x[..., 2:]], axis=-1)

    elif env=='cheetah':
        def obs_cost(obs):
            return -obs[..., 0]

        def act_cost(acs):
            return 0.1 * tf.reduce_sum(tf.square(acs), axis=-1)

        def obs_preproc(obs):
            return tf.concat([obs[..., 1:2], tf.sin(obs[..., 2:3]), 
                tf.cos(obs[..., 2:3]), obs[..., 3:]], axis=-1)

    elif env=='pusher':
        def obs_cost(obs):
            tip = obs[..., -9:-6]
            obj = obs[..., -6:-3]
            gol = obs[..., -3:]
            t_o = tf.reduce_sum(tf.abs(obj - tip), axis=-1)
            o_g = tf.reduce_sum(tf.abs(gol - obj), axis=-1)
            return .5*t_o + 1.25*o_g 

        def act_cost(acs):
            return 0.1 * tf.reduce_sum(tf.square(acs), axis=-1)

        def obs_preproc(obs):
            return obs 
    
    elif env=='reacher':
        def obs_cost(obs):
            return tf.reduce_sum(tf.square(obs[..., -3:]), axis=-1)

        def act_cost(acs):
            return 0.1 * tf.reduce_sum(tf.square(acs), axis=-1)

        def obs_preproc(obs):
            return obs 
    else:
        return None, None, None
    
    return obs_cost, act_cost, obs_preproc
