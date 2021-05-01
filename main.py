import os
import argparse
import json
import time
import numpy as np
from policy import Policy
from utils import *
from agent import Agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver
from tf_agents.utils import common
# Delete if no use
import tensorflow_probability as tfp

def main(args):
    # Load or create args
    if args.load_id is not '0':
        logpath = os.path.join(f'./experiments', args.load_id)
        args_filename = os.path.join(logpath, 'args.txt')
        args = load_args(args_filename)
        list_returns = list(np.load(os.path.join(logpath, 'returns.npy')))
    else:
        exp_id = str(int(time.time()))
        logpath = os.path.join(f'./experiments', exp_id)
        args_filename = os.path.join(logpath, 'args.txt')
        vars(args)['load_id']=exp_id
        list_returns = []

    filename = f'k{args.k}_ph{args.plan_hor}_ps{args.pop_size}_'\
               f'np{args.npart}_c{args.calibrate}'
    results_filename = os.path.join(logpath, 'returns.npy')
    
    if not(os.path.exists(logpath)):
        os.makedirs(logpath)

    # Setup learning environments
    env, video_env = get_env(args.env)
    eval_env, _ = get_env(args.env)
    
    policy = Policy(env.time_step_spec(), env.action_spec(), None, 
            args.model, args.num_nets, args.k, args.plan_hor, args.npart,
            args.pop_size, args.ts_sampler, args.a_sampler, args.env, args.calibrate)
    
    global_step = tf.compat.v1.train.get_or_create_global_step()
    agent = Agent(policy, train_step_counter=global_step)
    agent.initialize()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                        agent.collect_data_spec,
                        batch_size=1,
                        max_length=1000)

    replay_observer = [replay_buffer.add_batch]

    collect_op = dynamic_step_driver.DynamicStepDriver(
      env,
      agent.collect_policy,
      observers=replay_observer,
      num_steps=args.plan_hor)

    # Checkpointer
    checkpoint_dir = os.path.join(logpath, 'checkpoint')
    train_checkpointer = common.Checkpointer(
        ckpt_dir=checkpoint_dir,
        max_to_keep=1,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
        global_step=global_step
    )

    # Reload model
    if args.load_id is not '0':
        print('Restoring model with exp_id: {args.load_id}')
        train_checkpointer.initialize_or_restore()
        global_step = tf.compat.v1.train.get_global_step()

    # Initial data collection
    for i in range(args.ninit_rollouts):
        collect_op.run()

    # Train the agent and evaluate results
    eval_obs = eval_env.reset()
    for i in range(args.ntrain_iters):
        # Construct dataset
        collect_op.run()
        dataset = replay_buffer.as_dataset(
                        sample_batch_size=args.batch_size,
                        num_steps=args.plan_hor,
                        single_deterministic_pass=True)

        # Train the agent
        for e in range(args.epochs):
            for trajectories, _ in dataset:
                loss = agent.train(trajectories)
        
        # Collect evaluation data from the agent
        returns = 0
        for j in range(args.plan_hor):
            eval_obs = eval_env.step(policy.action(eval_obs))
            returns += eval_obs.reward
        list_returns += [returns.numpy()]
        print(f"train iter {i} Avg.Return:{np.mean(list_returns)}")

        # Save checkpoint
        if (i+1)%int(args.save_period) == 0:
            print("Saving model")
            train_checkpointer.save(global_step)

    #Save results
    np.save(results_filename, np.array(list_returns))

    #Save args
    with open(args_filename, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    print(f'Expected reward of {args.k}_{args.calibrate}({args.load_id}) on {args.env} '\
          f'after {agent.train_step_counter.numpy()} interactions: {np.mean(list_returns):.3f}')

    # Update results database
    with open('./experiments/exp_list.txt', 'a+') as f:
        f.write(f'{args.env}, {args.k}, {args.calibrate}, {args.num_nets}, {args.load_id}'\
                f', {agent.train_step_counter.numpy()}, {np.mean(list_returns):.3f}\n')
    
    # Video agent performance on environments
    if args.video:
        video_filename = os.path.join(logpath, f'{filename}.mp4')
        video_agent(env, video_env, policy, video_filename, args.frames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', type=str, default='cartpole',
                        help='Environment name: select from [cartpole, reacher, pusher, cheetah]')
    parser.add_argument('-model', type=str, default='p',
                        help='Model to estimate transition probabilities.')
    parser.add_argument('-calibrate',  dest='calibrate', action='store_true',
                        help='Enable calibration for the model')
    parser.add_argument('-no-calibrate',  dest='calibrate', action='store_false',
                        help='Disable calibration for the model')
    parser.set_defaults(calibrate=False)
    parser.add_argument('-logdir', type=str, default='log',
                        help='Directory to which results will be logged (default: ./log)')
    # Training Parameters
    parser.add_argument('-ntrain_iters', type=int, default=50,
                        help='Number of training iterations. If load, this will \
                              be the number of iterations to continue training the model')
    parser.add_argument('-plan_hor', type=int, default=7,
                        help='Length of trajectories to sample.')
    parser.add_argument('-batch_size', type=int, default=3,
                        help='Batch size will be multiplied by plan_hor')
    parser.add_argument('-num_rollout_per_iter', type=int, default=1,
                        help='Number of env. interactions per train step.')
    parser.add_argument('-ninit_rollouts', type=int, default=5,
                        help='Number of initial rollouts to collect')
    parser.add_argument('-epochs', type=int, default=5,
                        help='Number of epochs')
    # MPC Parameters
    parser.add_argument('-pop_size', type=int, default=10,
                        help='Number of trajectories to sample.')
    parser.add_argument('-npart', type=int, default=10,
                        help='Number of particles. Must be multiple of num_nets')
    parser.add_argument('-a_sampler', type=str, default='random',
                        help='Action sampler.')
    parser.add_argument('-ts_sampler', type=str, default='ts1',
                        help='Trajectory sampler.')
    # Evaluation Parameters
    parser.add_argument('-video', action='store_true',
                        help='Video agent interacting with environment.')
    parser.add_argument('-frames', type=int, default=500,
                        help='Frames to video the agent for.')
    # Model Parameters
    parser.add_argument('-num_nets', type=int, default=5,
                        help='Number of networks in the ensemble.')
    parser.add_argument('-k', type=int, default=3,
                        help='Number of Gaussian Mixture Components')
    # Checkpoint parameters
    parser.add_argument('-load_id', type=str, default='0',
            help='Experiment to load.')
    parser.add_argument('-save_period', type=int, default=25,
            help='Save model period')
    # Agent Paramters
    args = parser.parse_args()

    main(args)

