# Simulate training for both agents for one episode
# Originally written for agents of the class DQNAgent.

def train_dyad(env, agent1, agent2):
    #required imports: none
    old_observations = env.reset(renew_traj=True); #old_observations = np.asarray(observations)
    f1 = agent1.get_force(old_observations, eps=1.)
    f2 = agent2.get_force(old_observations, eps=1.)

    while True:
        t = env.get_time()
        observations, reward, done, _ = env.step([f1, f2])
               
        utility1 = agent1.compute_utility(reward, f1)
        utility2 = agent2.compute_utility(reward, f2)
        
        # Add the experience
        err = observations[0]-observations[2]
        if abs(err)> env.max_err*agent1.buffer.tag:
            agent1.add_experience((old_observations, f1, observations, utility1))
            agent2.add_experience((old_observations, f2, observations, utility2))
            
        # Take one training step for each agent
        agent1.train_step()
        agent2.train_step()

        # Generate action for next env interaction
        f1 = agent1.get_force(observations, eps=1.)
        f2 = agent2.get_force(observations, eps=1.)
        old_observations = observations

        if done is True:
            break
    return agent1, agent2