package apple_lib.environment;

import apple_lib.lp.GaussianElimination;
import apple_lib.lp.LP_Solver;
import apple_lib.lp.TwoPhaseSimplex;

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
			// Bellman update
			double[] update = new double[sim.S];
			for(int state = 0; state < sim.S; state++) {
				// Store candidates to perform max
				double[] candidates = new double[sim.A];
				for(int action = 0; action < sim.A; action++) {
					for(int next = 0; next < sim.S; next++) {
						candidates[action] += sim.T(state, action, next) * (sim.r(state, action, next) + sim.gamma * value[next]);
					}
				}
				// Determine maximum
				update[state] = candidates[0];
				for(int action = 1; action < sim.A; action++) {
					update[state] = Math.max(update[state], candidates[action]);
				}
			}
			value = update;
		}

		return policy(value);
	}

	/**
	 * Uses linear programming to solve the primal problem
	 */
	public int[] primal_lp() {
		LP_Solver solver = new TwoPhaseSimplex(sim.S);

		// Iterate through each state and make the constraint and objective
		double[] objective = new double[sim.S];
		for(int state = 0; state < sim.S; state++) {
			objective[state] = sim.mu(state);
			for(int action = 0; action < sim.A; action++) {
				double[] constraint = new double[sim.S + 1];
				constraint[state] = -1;
				for(int next = 0; next < sim.S; next++) {
					double prob = sim.T(state, action, next);
					constraint[next] += sim.gamma * prob;
					constraint[sim.S] -= prob * sim.r(state, action, next);
				}
				solver.subject_to(constraint);
			}
		}
		solver.minimize(objective);

		// Solve and extract policy
		if(solver.solve()) {
			double[] value = solver.parameters();
			return policy(value);
		}

		throw new RuntimeException(String.format(
			"Solver failed (%s)", solver.is_infeasible() ? "Infeasible" : solver.is_unbounded() ? "Unbounded" : "N/A"
		));
	}

	/**
	 * Uses linear programming to solve the dual problem
	 */
	public int[] dual_lp() {
		LP_Solver solver = new TwoPhaseSimplex(sim.S * sim.A);

		// Define objective function
		double[] objective = new double[sim.S * sim.A];
		for(int state = 0; state < sim.S; state++) {
			for(int action = 0; action < sim.A; action++) {
				for(int next = 0; next < sim.S; next++) {
					int index = state * sim.A + action;
					objective[index] += sim.T(state, action, next) * sim.r(state, action, next);
				}
			}
		}
		solver.maximize(objective);

		// Iterate through each state and make a constraint
		for(int state = 0; state < sim.S; state++) {
			double[] constraint = new double[sim.S * sim.A + 1];

			// Marginal state distribution
			for(int action = 0; action < sim.A; action++) {
				int index = state * sim.A + action;
				constraint[index] += 1;
			}

			// Initialization probability
			constraint[sim.S * sim.A] += (1 - sim.gamma) * sim.mu(state);

			// Transition probability
			for(int action = 0; action < sim.A; action++) {
				int index = state * sim.A + action;
				for(int next = 0; next < sim.S; next++) {
					constraint[index] -= sim.gamma * sim.T(state, action, next);
				}
			}
			
			solver.subject_to(constraint);
		}

		// Solve and extract policy
		if(solver.solve()) {
			int[] policy = new int[sim.S];
			double[] d = solver.parameters();
			// Iterate through each state
			for(int state = 0; state < sim.S; state++) {
				// Determine how many actions have non-zero probability
				int count = 0;
				for(int action = 0; action < sim.A; action++) {
					int index = state * sim.A + action;
					if(d[index] > 0) count++;
				}
				// Pick one
				count = (int) (count * Math.random());
				for(int action = 0; action < sim.A; action++) {
					int index = state * sim.A + action;
					if(d[index] > 0) {
						if(count == 0) {
							policy[state] = action;
							break;
						}
						count--;
					}
				}
			}

			return policy;
		}

		throw new RuntimeException(String.format(
			"Solver failed (%s)", solver.is_infeasible() ? "Infeasible" : solver.is_unbounded() ? "Unbounded" : "N/A"
		));
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
					if(count == 0) {
						out[state] = i;
						break;
					}
					count--;
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

		// Reduce to reduced row echilon form
		GaussianElimination solver = new GaussianElimination(sim.S, 2 * sim.S);
		solver.load(inverse_calc);
		inverse_calc = solver.rref();

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


