package apple_lib.environment;

import apple_lib.lp.GaussianElimination;

/**
 * Solves fully observable, stochastic, Markov games
 */
public class MarkovGameSolver {

	////////////////////////////////// FIELDS //////////////////////////////////

	/* Simulator */
	private MarkovGame sim;

	/////////////////////////////// CONSTRUCTORS ///////////////////////////////

	/**
	 * Basic constructor
	 */
	public MarkovGameSolver(MarkovGame game) {
		sim = game;
	}

	////////////////////////////////// METHODS /////////////////////////////////

	/**
	 * Solves the Markov game using policy iteration
	 */
	public double[][][] policy_iteration(int K, int N) {
		double[][][] policy = new double[sim.S][sim.I][];
		for(int state = 0; state < sim.S; state++) {
			for(int player = 0; player < sim.I; player++) {
				policy[state][player] = new double[sim.A[player]];
				for(int action = 0; action < sim.A[player]; action++) {
					policy[state][player][action] = (double) 1 / sim.A[player];
				}
			}
		}

		for(int iteration = 0; iteration < K; iteration++) {
			double[][] value = value(policy);
			double[][][] update = new double[sim.S][][];
			for(int state = 0; state < sim.S; state++) {
				int state_ = state;
				NormalFormGame normal = new NormalFormGame(sim.A) {
					public double r(int player, int... actions) {
						double reward = 0;
						for(int next = 0; next < sim.S; next++) {
							reward += sim.T(state_, next, actions) * (sim.r(state_, next, player, actions) + sim.gamma * value[next][player]);
						}
						return reward;
					}
				};
				NormalFormGameSolver solver = new NormalFormGameSolver(normal);
				update[state] = solver.fictitious_play(N);
			}
			policy = update;
		}

		return policy;
	}

	/**
	 * Determines the expected value of the full game, given the value of each state
	 */
	public double[] value(double[][] value) {
		double[] out = new double[sim.I];
		for(int state = 0; state < sim.S; state++) {
			double prob = sim.mu(state);
			for(int player = 0; player < sim.I; player++) {
				out[player] += prob * value[state][player];
			}
		}
		return out;
	}

	/**
	 * Determines the value function conditioned on the provided player policies
	 */
	public double[][] value(double[][][] policy) {
		// Determine transition and reward matrices
		double[][] transition = new double[sim.S][sim.S];
		double[][] reward = new double[sim.S][sim.I];
		for(int state = 0; state < sim.S; state++) {
			int[] actions = new int[sim.S];
			while(actions[sim.I - 1] < sim.A[sim.I - 1]) {
				double action_prob = 1;
				for(int player = 0; player < sim.I; player++) {
					action_prob *= policy[state][player][actions[player]];
				}
				for(int next = 0; next < sim.S; next++) {
					double prob = action_prob * sim.T(state, next, actions);
					transition[state][next] += prob;
					for(int player = 0; player < sim.I; player++) {
						reward[state][player] += prob * sim.r(state, next, player, actions);
					}
				}

				actions[0]++;
				for(int player = 0; player < sim.I - 1 && actions[player] >= sim.A[player]; player++) {
					actions[player] = 0;
					actions[player + 1]++;
				}
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
		double[][] value = new double[sim.S][sim.I];
		for(int state = 0; state < sim.S; state++) {
			for(int next = 0; next < sim.S; next++) {
				for(int player = 0; player < sim.I; player++) {
					value[state][player] += inverse[state][next] * reward[next][player];
				}
			}
		}

		return value;
	}

}

