def value_iteration(m, number_iterations: int = 1000) -> tuple[dict[str:str],dict[str,float]]:
    """
    A general value iteration implementation.
    Inputs:
        - class m representing Markov Decision Process.
        - number_iterations: defaults to 1000.
    Outputs:
        - pi: a dictionary containing an approximate optimal policy 
        e.g. {particular_state: particular_action}
        - v: a dictionary of the approximate value function, ordered by state. 
    Notes on implementation: 
        - v is by default initialised as 0 for every state.
        - The implementation does not store v as an array.
        Rather, it updates dictionaries for q and v at each iteration,
        so differs from pseudocode in (Poole and Mackworth, 2017).
        - Uses native pythonic data-structures to minimise dependencies.
    """
    states = m.states
    actions = m.actions
    state_action_pairs = m.create_pairs(m.states, m.actions)
    dynamics = m.dynamics(m.states, m.actions, m.probabilities)
    rewards = m.rewards(m.states, m.actions, m.reward_values)
    gamma = m.discount_factor
    
    #initialising 
    q = {key : 0 for key in state_action_pairs}
    v = {key: 0 for key in states}
    
    for i in range(number_iterations):
        #update Q
        for j in range(len(q)):
            probabilities = dict(filter(lambda item: item[0][1] in
                                        [state_action_pairs[j]], dynamics.items()))
            expected_value = sum([ list(probabilities.values())[k]*list(v.values())[k]
                                  for k in range(len(states))
                                 ])
            discounted_expected_value = gamma * expected_value
            reward = list(rewards.values())[j]
            q_update = {state_action_pairs[j] : reward + discounted_expected_value}
            q.update(q_update)
        #update V
        for k in range(len(v)):
            q_values = dict(filter(lambda item: item[0][0] in [states[k]], q.items()))
            maximum_value = max(list(q_values.values()))
            v_update = {states[k]:maximum_value}
            v.update(v_update)
    # Make policy pi
    pi = {key: None for key in states}
    for i in range(len(states)):
        #filtered list of q values for a fixed state
        q_list = list(dict(filter(lambda item: item[0][0] in [states[i]], q.items())).values())
        #action corresponding to value function at that state
        policy = actions[q_list.index(list(v.values())[i])]
        pi_update = { states[i]: policy}
        pi.update(pi_update)
    return pi, v


    