
# Reinforcement Learning Models

## Markov Decision Processes

To define an MDP, create a subclass of `MarkovDecisionProcess` and implement the reward function, transition matrix, and initial
state distributions. For a fixed parameter, these three methods should always return the same values. The constructor of the MDP
defines the state set, action set, and discount factor. States and actions are represented with integers starting from zero and
counting up. The validity of the class definition can be checked using the provided check method. 

    MarkovDecisionProcess mdp = ...;
    if(!MarkovDecisionProcess.check(mdp, probability_tolerance, min_reward, max_reward)) throw new RuntimeException();

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

    int action_taken = policy[state];

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

## Finite Horizon Markov Decision Process

An alternative MDP is the finite horizon setting. This setting does not include a discount factor, but maintains a finite
horizon. The transition and reward functions become time dependent. 

    FiniteHorizonMarkovDecisionProcess mdp = ...;
    if(!FiniteHorizonMarkovDecisionProcess.check(mdp, probability_tolerance, min_reward, max_reward)) throw new RuntimeException();

### Solving a Finite Horizon MDP

Solving a finite horizon MDP uses backward induction. The policy provided by the solvers are deterministic. Policies are
represented by two-dimensional integer arrays, where the time step and state are used as indices. 

    FiniteHorizonMarkovDecisionProcessSolver solver = new FiniteHorizonMarkovDecisionProcess(mdp);
    
    int[][] policy = solver.backward_induction();

    int action_taken = policy[time_step][state];

## Multi-Armed Bandit

This library also includes an interface for multi-armed bandits. 

    MultiArmedBandit bandit = ...;

The `MultiArmedBandit` class contains a static method that takes an empirical estimate for the expected reward produced by each
action. To use this tool, provide the bandit and the number of samples per action. Users should beware of numerical stability in
cases where a very large sample size is used. 

    double[] mu = MultiArmedBandit.mu(bandit, 128);

### Solving a Multi-Armed Bandit

When creating a solver for multi-armed bandits, the user may provide the expected reward for each action if these expectations
are known. Only a subset of algorithms provided will use these expectations. Providing the expected rewards is not required. If
they are not provided but are required by a method call, the empirical estimations will be used. Users can optionally set the
sample size for the empirical estimations. 

    // Providing the true expected rewards for each action
    MultiArmedBanditSolver solver_1 = new MultiArmedBanditSolver(bandit, mu);

    // Setting the sample size for an empirical estimate
    MultiArmedBanditSolver solver_2 = new MultiArmedBanditSolver(bandit, 128);

    // Using the default sample size
    MultiArmedBanditSolver solver_3 = new MultiArmedBanditSolver(bandit);

The solver uses the expected rewards for each action to determine the optimal action. 

    int optimal = solver.optimal_action();

More interestingly, the solver can apply the UCB algorithm. It will return the actions taken at each iteration as an array of
integers. To use the UCB algorithm, users must provide the iteration count and the scale factor for exploration. 

    int[] history = solver.ucb(interations, exploration_weight);

Once the trajectory has been created, the solver can calculate the regret for each action. There are also options to determine
the average and total regret. These three methods take an array of integers as the trajectory and return an array of doubles. 

    // Per action regret
    double[] regret = solver.total_regret(history);

    // Summation of regret for all past actions at each point in the trajectory
    double[] total_regret = solver.total_regret(history);

    // Average regret of past actions at each point in the trajectory
    double[] avg_regret = solver.average_regret(history);

