package apple_lib.environment;

/**
 * Solver for finite horizon Markov decision processes
 */
public class FiniteHorizonMarkovDecisionProcessSolver {

	// FIELDS //
	
	/* Simulator */
	private FiniteHorizonMarkovDecisionProcess sim;

	// CONSTRUCTORS //
	
	/**
	 * Basic constructor
	 */
	public FiniteHorizonMarkovDecisionProcessSolver(FiniteHorizonMarkovDecisionProcess mdp) {
		sim = mdp;
	}

	// METHODS //
	
	/**
	 * Uses backward induction to find the optimal policy at each time step
	 */
	public int[][] backward_induction() {
		// Store the value for each state at the next time step
		double[] value = new double[sim.S];

		// Calculate policy
		int[][] policy = new int[sim.H][sim.S];

		// Backward induction
		for(int time = sim.H - 1; time >= 0; time--) {
			// Iterate through each state
			for(int state = 0; state < sim.S; state++) {
				// Find Q for each action
				double[] Q = new double[sim.A];
				for(int action = 0; action < sim.A; action++) {
					// Take expectation
					for(int next = 0; next < sim.S; next++) {
						Q[action] += sim.T(time, state, action, next) * (sim.r(time, state, action, next) + value[next]);
					}
				}

				// Take max and argmax of Q
				value[state] = Q[0];
				int argmax = 1;
				for(int action = 1; action < sim.A; action++) {
					if(Q[action] > value[state]) {
						argmax = 1;
						value[state] = Q[action];
					} else if(Q[action] == value[state]) {
						argmax++;
					}
				}
				int choice = (int) (Math.random() * argmax);
				for(int action = 0; action < sim.A; action++) {
					if(Q[action] == value[state]) {
						choice--;
						if(choice < 0) {
							policy[time][state] = action;
							break;
						}
					}
				}
			}
		}

		return policy;
	}

	/**
	 * Applies UCB to solve a finite horizon game without knowing full information about the MDP. Returns the actions taken to
	 * interact with the MDP. 
	 */
	public int[][] ucb_vi(int iterations) {
		int[][] history = new int[iterations][sim.H];

		return null;
	}

	/**
	 * Determines the value of each state at each time step for a specific policy
	 */
	public double[][] value(int[][] policy) {
		// Store value function
		double[][] value = new double[sim.H][sim.S];

		// Store value of next time step
		double[] next_value = new double[sim.S];

		// Backward induction
		for(int time = sim.H - 1; time >= 0; time--) {
			// Iterate through each state
			for(int state = 0; state < sim.S; state++) {
				// Take expectation
				for(int next = 0; next < sim.S; next++) {
					value[time][state] += sim.T(time, state, policy[time][state], next) * (sim.r(time, state, policy[time][state], next) + next_value[next]);
				}
			}

			// Update values for next interation
			next_value = value[time];
		}
		return value;
	}

	/**
	 * Determines the Q function paired with the given value function
	 */
	public double[][][] action_value(double[][] value) {
		// Store the Q function
		double[][][] Q = new double[sim.H][sim.S][sim.A];
		double[] next_value = new double[sim.S];

		// Iterate
		for(int time = 0; time < sim.H; time++) {
			for(int state = 0; state < sim.S; state++) {
				for(int action = 0; action < sim.A; action++) {
					// Take expectation
					for(int next = 0; next < sim.S; next++) {
						Q[time][state][action] += sim.T(time, state, action, next) * (sim.r(time, state, action, next) + next_value[next]);
					}
				}
			}
			next_value = value[time];
		}

		return Q;
	}

	/**
	 * Determines the value function paired with the given Q function and policy
	 */
	public double[][] value(double[][][] Q, int[][] policy) {
		double[][] value = new double[sim.H][sim.S];

		for(int time = 0; time < sim.H; time++) {
			for(int state = 0; state < sim.S; state++) {
				value[time][state] = Q[time][state][ policy[time][state] ];
			}
		}

		return value;
	}

	/**
	 * Determines the value of a full MDP, given the value function
	 */
	public double value(double[][] value) {
		double expectation = 0;
		for(int state = 0; state < sim.S; state++) {
			expectation += sim.mu(state) * value[0][state];
		}
		return expectation;
	}

}

