# This file is currently only a temporary sketch

def get_ftr_outcome_rms(env, actor1, actor2, n_episodes=1):
    n_steps_max = env._max_episode_steps
    
    ftrs = []
    outcomes = []

    for i_episode in range(n_episodes):
        ftr_i=[]; outcome_i = []
        
        observations = env.reset(renew_traj=True); old_observations = np.asarray(observations)
        f1 = actor1(observations, 0.) #initalize s_old, a_old
        f2 = actor2(observations, 0.)
        ftr_vec = observations+[f1, f2]
        ftr_i.append(ftr_vec)

        while True:
            t = env.get_time()
            observations, reward, done, _ = env.step([f1, f2])
            observations_arr = np.asarray(observations)
            outcome = observations_arr-old_observations
            outcome_i.append(outcome)
            
            f1 = actor1(observations, f1)
            f2 = actor2(observations, f2)
            old_observations = observations_arr
            ftr_vec =  observations+[f1, f2]

            if done is True:
                break
   
        ftrs.append(ndrms(np.asarray(ftr_i), axis=0))
        outcomes.append(ndrms(np.asarray(outcome_i), axis=0))
        
    ftrs_rms = np.mean(np.asarray(ftrs), axis=0)
    outcomes_rms = np.mean(np.asarray(outcomes), axis=0)
    return ftrs_rms, outcomes_rms

# In benchmark
#     _, outcomes_rms = get_ftr_outcome_rms(env, agent1, agent2, n_episodes=3)
#     outcomes_rms += 0.00001*(outcomes_rms==0)

# def dyad_eval(env, agent1, agent2, n_episodes=100, normalizer=True):
#     # Empirical Model Evaluation
#     # For assessing the performance of two RLAgent agents playing 
#     # on DyadSlider.
#     # Arguments:
#     # n_episodes: the number of episodes used for averaging the quality of the policy
#     # normalize: if True, the reward is normalized by the number of time steps.

#     #     all_loss = np.zeros((n_episodes, env.observation_space.shape[0]))
#     reward_vals = np.zeros(n_episodes)


#     for i_episode in range(n_episodes):
#         cum_reward = 0.
# #         ep_loss = torch.zeros(env.observation_space.shape[0])
#         observations = env.reset(renew_traj=True);# old_observations = np.asarray(observations)
#         f1 = agent1.get_force(observations, eps=0)
#         f2 = agent2.get_force(observations, eps=0)
# #         ftr_vec = observations+[f1, f2]

#         while True:
#             t = env.get_time()
#             observations, reward, done, _ = env.step([f1, f2])
#             cum_reward += reward
# #             cum_reward += agent1.compute_utility(reward, f1)#reward
# #             observations_arr = np.asarray(observations)
# #             outcome = observations_arr-old_observations
# #             ftr_tensor, outcome_tensor = torch.FloatTensor(ftr_vec), torch.from_numpy(outcome).float()
# #             prediction = model(ftr_tensor)
            
# #             ep_loss += abs(prediction-outcome_tensor)
            
#             f1 = agent1.get_force(observations, eps=0)
#             f2 = agent2.get_force(observations, eps=0)
            
# #             old_observations = observations_arr
# #             ftr_vec =  observations+[f1, f2]

#             if done is True:
#                 break
# #         ep_loss = ep_loss.detach().numpy() /t
# #         if normalizer is not None:
# #             all_loss[i_episode,:] = ep_loss /normalizer
# #         else:
# #             all_loss[i_episode,:] = ep_loss
#         if normalizer is True:
#             reward_vals[i_episode] = cum_reward /t
#         else:
#             reward_vals[i_episode] = cum_reward
        
#     return np.mean(reward_vals, axis=0)