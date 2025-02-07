package apple_lib.environment;

public class MarkovGameSolver {

	private MarkovGame sim;

	public MarkovGameSolver(MarkovGame game) {
		sim = game;
	}

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

			for(int state = 0; state < sim.S; state++) {
				for(int player = 0; player < sim.I; player++) {
					System.out.print(String.format("%f ", value[state][player]));
				}
				System.out.println();
			}
			System.out.println();

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

	public double[][] value(double[][][] policy) {
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

		double[][] I_gT = new double[sim.S][sim.S];
		for(int state = 0; state < sim.S; state++) {
			for(int next = 0; next < sim.S; next++) {
				I_gT[state][next] = (state == next ? 1 : 0) - sim.gamma * transition[state][next];
			}
		}
		double[][] inverse_calc = new double[sim.S][2 * sim.S];
		for(int state = 0; state < sim.S; state++) {
			for(int next = 0; next < sim.S; next++) {
				inverse_calc[state][next] = I_gT[state][next];
				inverse_calc[state][next + sim.S] = state == next ? 1 : 0;
			}
		}
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
		for(int row = sim.S - 1; row >= 0; row--) {
			for(int target = 0; target < row; target++) {
				double base = inverse_calc[target][row];
				for(int col = row; col < 2 * sim.S; col++) {
					inverse_calc[target][col] -= inverse_calc[row][col] * base;
				}
			}
		}
		double[][] inverse = new double[sim.S][sim.S];
		for(int row = 0; row < sim.S; row++) {
			for(int col = 0; col < sim.S; col++) {
				inverse[row][col] = inverse_calc[row][col + sim.S];
			}
		}

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

