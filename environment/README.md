
# Reinforcement Learning Models

## Markov Decision Processes

To define an MDP, create a subclass of `MarkovDecisionProcess` and implement the reward function, transition matrix, and initial
state distributions. For a fixed parameter, these three methods should always return the same values. The constructor of the MDP
defines the state set, action set, and discount factor. States and actions are represented with integers starting from zero and
counting up. The validity of the class definition can be checked using the provided check method. 

    MarkovDecisionProcess mdp = ...;
    if(!MarkovDecisionProcess.check(mdp)) throw new RuntimeException();

### Solving an MDP

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

### Interacting with an MDP

There are two ways to interact with an MDP. A generator takes a state and action. It then uses an MDP object to determine the
next state and associated reward. 

    MarkovDecisionProcessGenerator gen = new MarkovDecisionProcessGenerator(mdp);
    gen.submit(curr_state, action);
    int next_state = gen.next_state();
    double reward = gen.reward();

The more realistic method is to use a simulator. A simulator uses an MDP object to generate a true trajectory. This is the 
normal way of interacting with an MDP. The full trajectory generated through the interaction can be retrieved from the
simulator. 

    MarkovDecisionProcessSimulator sim = new MarkovDecisionProcessSimulator(mdp);
    int state = sim.state();

    sim.submit(action);
    int new_state = sim.state();
    double reward = sim.reward();

    ...
    
    int[] state_trajectory = sim.state_trajectory();
    int[] action_trajectory = sim.action_trajectory();
    double[] reward_trajectory = sim.reward_trajectory();

If given a policy, a simulator can create a full trajectory of a given length. Pass deterministic policies as an array of
actions indexed by state, as given by the solvers. Using this method will not reset the stored trajectory before calculations. 

    // If desired, clear any values currently stored in the trajectory
    sim.reset();

    sim.simulate(policy, iteration_count);

    int[] state_trajectory = sim.state_trajectory();
    int[] action_trajectory = sim.action_trajectory();
    double[] reward_trajectory = sim.reward_trajectory();

