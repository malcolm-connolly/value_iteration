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
