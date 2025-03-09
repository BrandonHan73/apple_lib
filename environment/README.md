
# Reinforcement Learning Models

## Markov Decision Processes

To define an MDP, create a subclass of `MarkovDecisionProcess` and implement the reward function, transition matrix, and initial
state distributions. For a fixed parameter, these three methods should always return the same values. The constructor of the MDP
defines the state set, action set, and discount factor. States and actions are represented with integers starting from zero and
counting up. The validity of the class definition can be checked using the provided check method. 

    MarkovDecisionProcess mdp = ...;
    if(!MarkovDecisionProcess.check(mdp)) throw new RuntimeException();

Use the provided solver to get an optimal policy for the defined MDP. The policy is given by an array of integers. Use states
to index into the array. The elements of the array are actions that should be taken at each state. A resulting requirement is
that the solver calculates a deterministic policy. Four algorithms are provided. It should be noted that the LP solver is highly
unstable, so the LP algorithms for solving MDP's may not provide an accurate result. 

    MarkovDecisionProcessSolver solver = new MarkovDecisionProcessSolver(mdp);

    int[] pol_1 = solver.value_iteration(iterations);
    int[] pol_2 = solver.policy_iteration(iterations);
    int[] pol_3 = solver.primal_lp();
    int[] pol_4 = solver.dual_lp();

We can also determine the value function associated with a given policy. This can be useful for evaluating performance. 

    // Determine the value of each state, given the deterministic policy used
    double[] state_values = solver.value(policy);

    // Determine the expected value of the full MDP, given the value of each state
    double full_value = solver.value(state_values);

    // Determine an optimal policy, given the value of each state
    int[] opt_policy = solver.policy(state_values);

