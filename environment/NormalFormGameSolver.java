package apple_lib.environment;

import apple_lib.function.activation.SoftmaxFunction;

public class NormalFormGameSolver {

	private NormalFormGame sim;

	public NormalFormGameSolver(NormalFormGame game) {
		sim = game;
	}

	public double[][] fictitious_play(int K) {
		double[][] output = new double[sim.I][];
		int[][] empirical = new int[sim.I][];
		for(int player = 0; player < sim.I; player++) {
			output[player] = new double[sim.A[player]];
			empirical[player] = new int[sim.A[player]];

			for(int action = 0; action < sim.A[player]; action++) {
				output[player][action] = (double) 1 / sim.A[player];
			}
		}

		for(int iteration = 1; iteration <= K; iteration++) {
			for(int player = 0; player < sim.I; player++) {
				double[] expected_reward = new double[sim.A[player]];
				for(int action = 0; action < sim.A[player]; action++) {
					int[] actions = new int[sim.I];
					actions[player] = action;
					while(actions[sim.I - 1] < sim.A[sim.I - 1] && (player != sim.I - 1 || actions[player] == action)) {
						double reward = sim.r(player, actions);
						double prob = 1;
						for(int opponent = 0; opponent < sim.I; opponent++) {
							if(opponent == player) continue;
							prob *= output[opponent][actions[opponent]];
						}
						expected_reward[action] += reward * prob;
						actions[0]++;
						for(int update = 0; update < sim.I - 1; update++) {
							if(update == player && actions[update] != action) {
								actions[update] = action;
								actions[update + 1]++;
							} else if(actions[update] >= sim.A[update]) {
								actions[update] = 0;
								actions[update + 1]++;
							}
						}
					}
				}

				double max = expected_reward[0];
				for(int action = 0; action < sim.A[player]; action++) {
					max = Math.max(max, expected_reward[action]);
				}
				int count = 0;
				for(int action = 0; action < sim.A[player]; action++) {
					if(expected_reward[action] == max) count++;
				}
				int[] argmax = new int[count];
				for(int action = sim.A[player] - 1; action >= 0; action--) {
					if(expected_reward[action] == max) {
						count--;
						argmax[count] = action;
					}
				}
				int action = argmax[ (int) (argmax.length * Math.random()) ];
				empirical[player][action]++;
			}

			for(int player = 0; player < sim.I; player++) {
				for(int action = 0; action < sim.A[player]; action++) {
					output[player][action] = (double) empirical[player][action] / iteration;
				}
			}
		}

		return output;
	}

	public double[][] gradient_ascent(int K, double alpha) {
		double[][] output = new double[sim.I][];
		double[][] parameters = new double[sim.I][];
		for(int player = 0; player < sim.I; player++) {
			parameters[player] = new double[sim.A[player]];
			output[player] = SoftmaxFunction.implementation.pass(parameters[player]);
		}

		for(int iteration = 1; iteration <= K; iteration++) {
			for(int player = 0; player < sim.I; player++) {
				double[] gradient = new double[sim.A[player]];
				for(int action = 0; action < sim.A[player]; action++) {
					int[] actions = new int[sim.I];
					actions[player] = action;
					while(actions[sim.I - 1] < sim.A[sim.I - 1] && (player != sim.I - 1 || actions[player] == action)) {
						double reward = sim.r(player, actions);
						double prob = 1;
						for(int opponent = 0; opponent < sim.I; opponent++) {
							if(opponent == player) continue;
							prob *= output[opponent][actions[opponent]];
						}
						gradient[action] += reward * prob;
						actions[0]++;
						for(int update = 0; update < sim.I - 1; update++) {
							if(update == player && actions[update] != action) {
								actions[update] = action;
								actions[update + 1]++;
							} else if(actions[update] >= sim.A[update]) {
								actions[update] = 0;
								actions[update + 1]++;
							}
						}
					}
				}

				double[][] backpropogate = SoftmaxFunction.implementation.differentiate(parameters[player], output[player]);
				for(int action = 0; action < sim.A[player]; action++) {
					for(int deriv = 0; deriv < sim.A[player]; deriv++) {
						parameters[player][action] += alpha * backpropogate[deriv][action] * gradient[deriv];
					}
				}
			}

			for(int player = 0; player < sim.I; player++) {
				output[player] = SoftmaxFunction.implementation.pass(parameters[player]);
			}
		}

		return output;
	}

	public double[][] distribution_expectation(int K) {
		double[][] output = new double[sim.I][];
		double[][] empirical = new double[sim.I][];
		for(int player = 0; player < sim.I; player++) {
			output[player] = new double[sim.A[player]];
			empirical[player] = new double[sim.A[player]];

			for(int action = 0; action < sim.A[player]; action++) {
				output[player][action] = (double) 1 / sim.A[player];
			}
		}

		for(int iteration = 1; iteration <= K; iteration++) {
			for(int player = 0; player < sim.I; player++) {
				int[] actions = new int[sim.I];
				while(actions[sim.I - 1] < sim.A[sim.I - 1]) {
					double[] r = new double[sim.A[player]];
					for(int action = 0; action < sim.A[player]; action++) {
						actions[player] = action;
						r[action] = sim.r(player, actions);
					}
					double best = r[0];
					for(int action = 1; action < sim.A[player]; action++) {
						best = Math.max(best, r[action]);
					}
					int count = 0;
					for(int action = 0; action < sim.A[player]; action++) {
						if(r[action] == best) count++;
					}
					int[] argmax = new int[count];
					int index = 0;
					for(int action = 0; action < sim.A[player]; action++) {
						if(r[action] == best) {
							argmax[index] = action;
							index++;
						}
					}

					double prob = 1;
					for(int opponent = 0; opponent < sim.I; opponent++) {
						if(opponent == player) continue;
						prob *= output[opponent][actions[opponent]];
					}
					for(int update : argmax) {
						empirical[player][update] += prob / count;
					}

					actions[0]++;
					for(int update = 0; update < sim.I - 1; update++) {
						if(actions[update] >= sim.A[update]) {
							actions[update] = 0;
							actions[update + 1]++;
						}
					}

				}
			}

			for(int player = 0; player < sim.I; player++) {
				for(int action = 0; action < sim.A[player]; action++) {
					output[player][action] = empirical[player][action] / iteration;
				}
			}
		}

		return output;
	}

}

