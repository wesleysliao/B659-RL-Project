# Benchmark functions
# Originally written for agents from class DQNAgent
import numpy as np

def policy_ts(env, agent1, agent2):
    # Behavioral profile of the agents
    # Returns time series of events in an episode
    # Arguments:
    # normalize: if True, the reward is normalized by the number of time steps.
    eps=0; verbose=True
    
#         ep_loss = torch.zeros(env.observation_space.shape[0])
    observations = env.reset(renew_traj=True);# old_observations = np.asarray(observations)
    f1, q1vals = agent1.get_force(observations, eps=eps, verbose=verbose)
    f2, q2vals = agent2.get_force(observations, eps=eps, verbose=verbose)
#         ftr_vec = observations+[f1, f2]
    
    t_ts=[]
    r_ts, x_ts = [],[]
    f1_ts, f2_ts = [],[]
    u1_ts,u2_ts = [],[];  cum_reward = 0.
    q10_ts, q11_ts, q20_ts, q21_ts = [],[],[],[]
    
    while True:
        t = env.get_time()
        observations, reward, done, _ = env.step([f1, f2])
        
        t_ts.append(t)
        r_ts.append(observations[0]); x_ts.append(observations[2]) # Logging
        f1_ts.append(f1) # Logging; 
        f2_ts.append(f2) # Logging
        q10_ts.append(q1vals[0])
        q11_ts.append(q1vals[1])
        q20_ts.append(q2vals[0])
        q21_ts.append(q2vals[1])
    
        cum_reward += reward#reward
        u1_ts.append(agent1.compute_utility(reward, f1))
        u2_ts.append(agent2.compute_utility(reward, f2))#reward

        f1, q1vals = agent1.get_force(observations, eps=eps, verbose=verbose)
        f2, q2vals = agent2.get_force(observations, eps=eps, verbose=verbose)

        if done is True:
            break

    cum_reward = cum_reward /t
    a1_cum_reward = np.mean(u1_ts)
    a2_cum_reward = np.mean(u2_ts)
        
    return t_ts, (r_ts,x_ts), (f1_ts, f2_ts), (q10_ts, q11_ts, q20_ts, q21_ts), (u1_ts, u2_ts, cum_reward, a1_cum_reward, a2_cum_reward)

def dyad_eval(env, agent1, agent2, n_episodes=100, normalizer=True):
    # Empirical Model Evaluation
    # For assessing the performance of two RLAgent agents playing 
    # on DyadSlider.
    # Arguments:
    # n_episodes: the number of episodes used for averaging the quality of the policy
    # normalize: if True, the reward is normalized by the number of time steps.

    reward_vals = np.zeros(n_episodes)


    for i_episode in range(n_episodes):
        cum_reward = 0.
        observations = env.reset(renew_traj=True);
        f1 = agent1.get_force(observations, eps=0)
        f2 = agent2.get_force(observations, eps=0)
#         ftr_vec = observations+[f1, f2]

        while True:
            t = env.get_time()
            observations, reward, done, _ = env.step([f1, f2])
            cum_reward += reward
            
            f1 = agent1.get_force(observations, eps=0)
            f2 = agent2.get_force(observations, eps=0)
            
            if done is True:
                break
        if normalizer is True:
            reward_vals[i_episode] = cum_reward /t
        else:
            reward_vals[i_episode] = cum_reward
        
    return np.mean(reward_vals, axis=0)
    