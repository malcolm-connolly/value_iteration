class Markov_Decision_Process : 
    """
    A class which implements an MDP with discounting for use with value iteration.
    
    Variables:
    states: a list of strings representing states 
    actions: a list of strings representing actions
    reward_values: a list of floats representing reward values
    for each state-action pair, ordered by state.
    probabilities: a list of the transition probabilities to each target state
    given state-action pair ordered first by target state,
    then state of the state-action pair.
    discount_factor: defaults to 1.

    Methods:
    create_pairs(): creates a list of tuples of states and actions, (state, action)
    rewards():
        inputs: states, actions, reward_values
        outputs: dictionary whose keys are state-action pairs with values from reward_values
    dynamics(): 
        inputs:
        states, actions, probabilities
        to each target state given state-action pair ordered first by target state
        then state of the state-action pair.
        outputs:
        a dictionary whose values are transition probabilities and whose keys
        are tuples of target state and state-action pairs. 
    """
    states = None
    actions = None
    probabilities = None
    reward_values = None
    discount_factor = None
    def create_pairs(self, states: list[str],
                     actions: list[str]) -> list[tuple[str,str]]:
        return([(state, action) for state in self.states for action in self.actions])
    def dynamics(self, states: list[str],
                 actions: list[str],
                 probabilities: list[float]) -> dict[tuple: float]:
        state_action_pairs = self.create_pairs(self.states, self.actions)
        dynamics_dictionary = {key: None for key in 
                               [(target_state, pair) for target_state in self.states
                                for pair in state_action_pairs]}
        for i in range(len(dynamics_dictionary)):
            value_update = { list(dynamics_dictionary.keys())[i]: self.probabilities[i]}
            dynamics_dictionary.update(value_update)
        return dynamics_dictionary
      
    def rewards(self, states: list[str],actions: list[str],
                reward_values : list[float]) -> dict[tuple: float]:
        state_action_pairs = self.create_pairs(self.states, self.actions)
        rewards_dictionary = {key: None for key in state_action_pairs}
        for i in range(len(rewards_dictionary)):
            value_update = { list(rewards_dictionary.keys())[i]: self.reward_values[i]}
            rewards_dictionary.update(value_update)    
        return rewards_dictionary
    
    def __init__(self, states: list[str], actions: list[str],
                 probabilities: list[float], reward_values: list[float],
                 discount_factor: float = 1) -> None:
        self.states = states
        self.actions = actions
        self.probabilities = probabilities
        self.reward_values = reward_values
        self.discount_factor = discount_factor


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


    
