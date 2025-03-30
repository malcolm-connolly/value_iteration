# `value_iteration`
- [**A Python package implementing value iteration for Markov decision processes**](#)
- [Installation](#installation)
- [Example of use](#example-of-use)
  - [Two state example](#two-state-example)
- [GitHub Repository](#github-repository)
- [Author](#author)
- [References](#references)


## A Python package implementing value iteration for Markov decision processes

This package contains a function value_iteration() which implements a value iteration algorithm based on pseudocode in [(Poole and Mackworth, Figure 9.16; 2017)]. 

[(Poole and Mackworth, Figure 9.16; 2017)]: https://artint.info/2e/html2e/ArtInt2e.Ch9.S5.SS2.html



## Installation

#### Install from GitHub with pip

    python -m pip install "git+https://github.com/malcolm-connolly/value_iteration"

You may also need to prefix this with an exclamation mark.

    !python -m pip install "git+https://github.com/malcolm-connolly/value_iteration"

## Example of use

#### Two-state example

This example is 9.27 from [(Poole and Mackworth; 2017)]. The example describes Sam who is in one of two states (healthy or sick) and who must decide whether to take one of two actions (relax or party). 

[(Poole and Mackworth; 2017)]: https://artint.info/2e/html2e/ArtInt2e.Ch9.S5.html#Ch9.Thmciexamplered27

First create the class for Sam's dilemma. 

``` python
from value_iteration import value_iteration, Markov_Decision_Process

sam = Markov_Decision_Process(states = ['healthy', 'sick'],
                              actions = ['relax', 'party'],
                              probabilities = [0.95, 0.7, 0.5, 0.1, 0.05, 0.3, 0.5, 0.9],
                              reward_values = [7, 10, 0, 2],
                              discount_factor = 0.8)

output = value_iteration(sam)
output
```

    ({'healthy': 'party', 'sick': 'relax'},
    {'healthy': 35.71428571428571, 'sick': 23.80952380952381})

The solution to the dilemma is that when Sam is healthy he should party, and when he is sick he should relax. 
    

#### Further information on class implementation

In this implementation we create a class for a Markov decision process. On creating an instance of the Markov_Decision_Process class the user must specify four lists; a list of states; a list of actions; a list of the reward values for each state-action pair ordered by state; and, a list of the transition probabilities to each target state given state-action pair ordered first by target state then state of the state-action pair. 

The order of the reward values list can be thought of as the order one would read a matrix of these values indexed by states as rows and actions as columns, that is: from left to right in rows from top row to bottom row. The list is then a flattened version of this imagined matrix. The order of the transition probabilities is the same except there is an imagined matrix for each target state, and these are read in the order already imposed on the states.

For an example of how this order affects the dictionary internal to the Markov_Decision_Process class, see below.

``` python
sam.rewards(sam.states, sam.actions, sam.reward_values)
```

    {('healthy', 'relax'): 7,
    ('healthy', 'party'): 10,
    ('sick', 'relax'): 0,
    ('sick', 'party'): 2}

``` python
sam.dynamics(sam.states, sam.actions, sam.probabilities)
```

    {('healthy', ('healthy', 'relax')): 0.95,
     ('healthy', ('healthy', 'party')): 0.7,
     ('healthy', ('sick', 'relax')): 0.5,
     ('healthy', ('sick', 'party')): 0.1,
     ('sick', ('healthy', 'relax')): 0.05,
     ('sick', ('healthy', 'party')): 0.3,
     ('sick', ('sick', 'relax')): 0.5,
     ('sick', ('sick', 'party')): 0.9}


I chose to use the list structure so that the package is stand-alone using base pythonic constructs independent of other packages, for example numpy and its arrays, as this makes the code more reusable and resilient to external changes or deprecation. The user does not need to write out or specify all the state-action pairs or keys of the dynamics dictionary, as these are conveniently generated within the class. 


## GitHub Repository

There is a public repository at the following site <https://github.com/malcolm-connolly/value_iteration>

## Author

- Malcolm Connolly: [email](mailto:m.connolly4@lancaster.ac.uk) (**Author**)

## References

Poole, D. L., & Mackworth, A. K. (2017). Artificial Intelligence: Foundations of Computational Agents (Second edition.). Cambridge University Press. <https://doi.org/10.1017/9781108164085>
