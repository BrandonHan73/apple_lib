package apple_lib.lp;

/**
 * Basic framework for a linear optimizer
 */
public interface LP_Solver {

	/**
	 * Adds a constraint to the solver. Parameters represent the coefficients
	 * for each variable. The last parameter represents the constant. 
	 * ax + by + cz + ... <= K
	 */
	public void subject_to(double... params);

	/**
	 * Defines the objective function. The solver should attempt to maximize.
	 * Will override any past objective functions. Parameters represent the
	 * coefficients of each variable. 
	 */
	public void maximize(double... params);

	/**
	 * Defines the objective function. The solver should attempt to minimize.
	 * Will override any past objective functions. Parameters represent the
	 * coefficients of each variable. 
	 */
	public void minimize(double... params);

	/**
	 * Optimizes the current objective function subject to the given
	 * constraints. Returns true if a solution was found. 
	 */
	public boolean solve();

	/**
	 * To be used after solve. Returns true if the solver determined that no
	 * feasible solutions exist. 
	 */
	public boolean is_infeasible();

	/**
	 * To be used after solve. Returns true if the solver determined that the
	 * solution is unbounded. 
	 */
	public boolean is_unbounded();

	/**
	 * Returns the resulting output of the optimized objective function.
	 */
	public double value();

	/**
	 * Returns an optimal solution to the optimization problem. 
	 */
	public double[] parameters();

}

