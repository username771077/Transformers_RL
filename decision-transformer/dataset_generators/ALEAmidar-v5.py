import h5py
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
import ale_py

env = gym.make("ALE/Amidar-v5")
env = RecordEpisodeStatistics(env)

buffer_size = 1000 
buffer_actions = []
buffer_observations = []
buffer_rewards = []
buffer_terminals = []
buffer_timeouts = []
buffer_lives = []
buffer_episode_frame_number = []
buffer_frame_number = []

obs_shape = env.observation_space.shape
action_dtype = np.int64  

with h5py.File('Amidar-v5.h5', 'w') as f:
    d_actions = f.create_dataset('actions', 
                                 shape=(0,), 
                                 maxshape=(None,), 
                                 dtype=action_dtype,
                                 compression='gzip', 
                                 chunks=True)
    d_observations = f.create_dataset('observations', 
                                      shape=(0,)+obs_shape, 
                                      maxshape=(None,)+obs_shape, 
                                      dtype=np.uint8, 
                                      compression='gzip', 
                                      chunks=True)
    d_rewards = f.create_dataset('rewards', 
                                 shape=(0,), 
                                 maxshape=(None,), 
                                 dtype=np.float32,
                                 compression='gzip', 
                                 chunks=True)
    d_terminals = f.create_dataset('terminals', 
                                   shape=(0,), 
                                   maxshape=(None,), 
                                   dtype=bool,
                                   compression='gzip', 
                                   chunks=True)
    d_timeouts = f.create_dataset('timeouts', 
                                  shape=(0,), 
                                  maxshape=(None,), 
                                  dtype=bool,
                                  compression='gzip', 
                                  chunks=True)

    infos_group = f.create_group('infos')
    d_lives = infos_group.create_dataset('lives', shape=(0,), maxshape=(None,), dtype=np.int32, compression='gzip', chunks=True)
    d_episode_frame_number = infos_group.create_dataset('episode_frame_number', shape=(0,), maxshape=(None,), dtype=np.int32, compression='gzip', chunks=True)
    d_frame_number = infos_group.create_dataset('frame_number', shape=(0,), maxshape=(None,), dtype=np.int32, compression='gzip', chunks=True)

    def flush_to_disk():
        if len(buffer_actions) == 0:
            return

        actions_arr = np.array(buffer_actions, dtype=action_dtype)
        observations_arr = np.array(buffer_observations, dtype=np.uint8)
        rewards_arr = np.array(buffer_rewards, dtype=np.float32)
        terminals_arr = np.array(buffer_terminals, dtype=bool)
        timeouts_arr = np.array(buffer_timeouts, dtype=bool)
        lives_arr = np.array(buffer_lives, dtype=np.int32)
        efn_arr = np.array(buffer_episode_frame_number, dtype=np.int32)
        fn_arr = np.array(buffer_frame_number, dtype=np.int32)

        current_size = d_actions.shape[0]
        new_size = current_size + len(actions_arr)

        d_actions.resize((new_size,))
        d_actions[current_size:new_size] = actions_arr

        d_observations.resize((new_size,)+obs_shape)
        d_observations[current_size:new_size] = observations_arr

        d_rewards.resize((new_size,))
        d_rewards[current_size:new_size] = rewards_arr

        d_terminals.resize((new_size,))
        d_terminals[current_size:new_size] = terminals_arr

        d_timeouts.resize((new_size,))
        d_timeouts[current_size:new_size] = timeouts_arr

        d_lives.resize((new_size,))
        d_lives[current_size:new_size] = lives_arr

        d_episode_frame_number.resize((new_size,))
        d_episode_frame_number[current_size:new_size] = efn_arr

        d_frame_number.resize((new_size,))
        d_frame_number[current_size:new_size] = fn_arr

        buffer_actions.clear()
        buffer_observations.clear()
        buffer_rewards.clear()
        buffer_terminals.clear()
        buffer_timeouts.clear()
        buffer_lives.clear()
        buffer_episode_frame_number.clear()
        buffer_frame_number.clear()

    total_steps = 0
    for episode in range(10000):
        observation, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_observation, reward, terminated, truncated, info = env.step(action)

            buffer_actions.append(action)
            buffer_observations.append(next_observation)
            buffer_rewards.append(reward)
            buffer_terminals.append(terminated)
            buffer_timeouts.append(truncated)
            buffer_lives.append(info['lives'])
            buffer_episode_frame_number.append(info['episode_frame_number'])
            buffer_frame_number.append(info['frame_number'])

            done = terminated or truncated
            total_steps += 1

            if len(buffer_actions) >= buffer_size:
                flush_to_disk()
    flush_to_disk()

env.close()
