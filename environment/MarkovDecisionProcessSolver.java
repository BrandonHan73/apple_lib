package apple_lib.environment;

/**
 * Solver for Markov decision processes
 */
public class MarkovDecisionProcessSolver {

	////////////////////////////////// FIELDS //////////////////////////////////

	/* Simulator */
	private MarkovDecisionProcess sim;

	/////////////////////////////// CONSTRUCTORS ///////////////////////////////

	/**
	 * Basic constructor
	 */
	public MarkovDecisionProcessSolver(MarkovDecisionProcess game) {
		sim = game;
	}

	////////////////////////////////// METHODS /////////////////////////////////

	/**
	 * Solves the Markov decision process using policy iteration
	 */
	public int[] policy_iteration(int K) {
		// Initialize policy randomly
		int[] policy = new int[sim.S];
		for(int state = 0; state < sim.S; state++) {
			policy[state] = (int) (sim.A * Math.random());
		}

		// Perform the specified number of iterations
		for(int iteration = 0; iteration < K; iteration++) {
			double[] value = value(policy);
			policy = policy(value);
		}

		return policy;
	}

	/**
	 * Solves the Markov decision process using value iteration
	 */
	public int[] value_iteration(int K) {
		// Initialize a zero value function
		double[] value = new double[sim.S];
		for(int iteration = 0; iteration < K; iteration++) {
			// Determine optimal policy of the current value function
			int[] policy = policy(value);
			// Bellman update
			double[] update = new double[sim.S];
			for(int state = 0; state < sim.S; state++) {
				for(int next = 0; next < sim.A; next++) {
					update[state] += sim.T(state, policy[state], next) * (sim.r(state, policy[state], next) + sim.gamma * value[next]);
				}
			}
			value = update;
		}
		return policy(value);
	}

	/**
	 * Creates a deterministic policy that maximizes the expected reward given a value function. 
	 */
	public int[] policy(double[] value) {
		int[] out = new int[sim.S];
		for(int state = 0; state < sim.S; state++) {
			// Determine the expected reward for each action
			double[] reward = new double[sim.A];
			for(int action = 0; action < sim.A; action++) {
				reward[action] = 0;
				for(int next = 0; next < sim.S; next++) {
					reward[action] += sim.T(state, action, next) * (sim.r(state, action, next) + sim.gamma * value[next]);
				}
			}
			// Take argmax
			double max = reward[0];
			int count = 1;
			for(int i = 1; i < sim.A; i++) {
				if(reward[i] > max) {
					max = reward[i];
					count = 1;
				} else if(reward[i] == max) {
					count++;
				}
			}
			count = (int) (count * Math.random());
			for(int i = 0; i < sim.A; i++) {
				if(reward[i] == max) {
					count--;
					if(count == 0) {
						out[state] = i;
						break;
					}
				}
			}
		}
		return out;
	}

	/**
	 * Determines the expected value of the full game, given the value of each state
	 */
	public double value(double[] value) {
		double out = 0;
		for(int state = 0; state < sim.S; state++) {
			out += sim.mu(state) * value[state];
		}
		return out;
	}

	/**
	 * Determines the value function conditioned on the provided player policy
	 */
	public double[] value(int[] policy) {
		// Determine transition and reward matrices
		double[][] transition = new double[sim.S][sim.S];
		double[] reward = new double[sim.S];
		for(int state = 0; state < sim.S; state++) {
			// Take expectation of next state and reward given current state and player action
			for(int next = 0; next < sim.S; next++) {
				double prob = sim.T(state, policy[state], next);
				transition[state][next] += prob;
				reward[state] += prob * sim.r(state, policy[state], next);
			}
		}

		// Create matrix to take the inverse of
		double[][] I_gT = new double[sim.S][sim.S];
		for(int state = 0; state < sim.S; state++) {
			for(int next = 0; next < sim.S; next++) {
				I_gT[state][next] = (state == next ? 1 : 0) - sim.gamma * transition[state][next];
			}
		}
		// Set up matrix used for inverse calculation by appending identity matrix
		double[][] inverse_calc = new double[sim.S][2 * sim.S];
		for(int state = 0; state < sim.S; state++) {
			for(int next = 0; next < sim.S; next++) {
				inverse_calc[state][next] = I_gT[state][next];
				inverse_calc[state][next + sim.S] = state == next ? 1 : 0;
			}
		}
		// First pass for row echilon form
		for(int row = 0; row < sim.S; row++) {
			double base = inverse_calc[row][row];
			if(base == 0) throw new RuntimeException();
			for(int col = 0; col < 2 * sim.S; col++) {
				inverse_calc[row][col] /= base;
			}
			for(int target = row + 1; target < sim.S; target++) {
				base = inverse_calc[target][row];
				for(int col = 0; col < 2 * sim.S; col++) {
					inverse_calc[target][col] -= inverse_calc[row][col] * base;
				}
			}
		}
		// Second pass for reduced row echilon form
		for(int row = sim.S - 1; row >= 0; row--) {
			for(int target = 0; target < row; target++) {
				double base = inverse_calc[target][row];
				for(int col = row; col < 2 * sim.S; col++) {
					inverse_calc[target][col] -= inverse_calc[row][col] * base;
				}
			}
		}
		// Extract inverse matrix
		double[][] inverse = new double[sim.S][sim.S];
		for(int row = 0; row < sim.S; row++) {
			for(int col = 0; col < sim.S; col++) {
				inverse[row][col] = inverse_calc[row][col + sim.S];
			}
		}

		// Perform matrix multiplication
		double[] value = new double[sim.S];
		for(int state = 0; state < sim.S; state++) {
			for(int next = 0; next < sim.S; next++) {
				value[state] += inverse[state][next] * reward[next];
			}
		}

		return value;
	}

}


