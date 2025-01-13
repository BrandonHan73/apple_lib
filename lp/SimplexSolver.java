package apple_lib.lp;

import java.util.ArrayList;
import java.util.List;

/**
 * Bare bones simplex method as introduced in Linear Programming by Vasek
 * Chvatal. 
 *
 * Assumes the following. 
 *  - Origin is a feasible solution
 *  - No cycles occur
 */
public class SimplexSolver implements LP_Solver {

	////////////////////////////////// FIELDS //////////////////////////////////
	
	/* Total number of variable to tune */
	public final int variable_count;

	/* Total number of constraints */
	private int constraint_count;

	/* Constraints */
	private List<double[]> constraints;

	/* Objective function. Assumes maximization. */
	private double[] objective;

	/* Distinguishes minimization from maximization */
	private boolean minimize;

	/* Stores output for collection. Assumes maximization. */
	private boolean infeasible, unbounded;
	private double[][] params;
	private double output;

	/* Dictionary used for calculations */
	private int dictionary_rows, dictionary_cols;
	private double[][] dictionary;

	/////////////////////////////// CONSTRUCTORS ///////////////////////////////

	/**
	 * Basic constructor. Restricts variable count. 
	 */
	public SimplexSolver(int variables) {
		variable_count = variables;

		constraint_count = 0;
		constraints = new ArrayList<>();

		objective = new double[variable_count];
		minimize = false;

		infeasible = false;
		unbounded = false;
		params = null;
		output = 0;

		dictionary = null;
		dictionary_rows = -1;
		dictionary_cols = -1;
	}

	////////////////////////////////// METHODS /////////////////////////////////

	/**
	 * Performs a pivot. 
	 */
	private void pivot(int enter, int exit) {
		if(dictionary == null) {
			throw new RuntimeException("Attempted to pivot while not solving a problem");
		}
		if(enter < 0 || dictionary_cols <= enter) {
			throw new RuntimeException(String.format("Entering index %d is invalid for %d columns", enter, dictionary_cols));
		}
		if(exit < 0 || dictionary_rows <= exit) {
			throw new RuntimeException(String.format("Exiting index %d is invalid for %d rows", exit, dictionary_rows));
		}

		double pivot_value = dictionary[exit][enter];
		if(pivot_value == 0) {
			throw new RuntimeException("Cannot use zero for pivot value");
		}
		for(int col = 0; col < dictionary_cols; col++) {
			dictionary[exit][col] /= pivot_value;
		}

		for(int row = 0; row < dictionary_rows; row++) {
			if(row == exit) continue;

			pivot_value = dictionary[row][enter];
			for(int col = 0; col < dictionary_cols; col++) {
				dictionary[row][col] -= dictionary[exit][col] * pivot_value;
			}
		}
	}

	/**
	 * Initializes dictionary variable based on the current constraints and
	 * objective
	 */
	private void load_dictionary() {
		dictionary_rows = constraint_count + 1;
		dictionary_cols = variable_count + constraint_count + 1;

		dictionary = new double[dictionary_rows][dictionary_cols];
		
		for(int con = 0; con < constraint_count; con++) {
			double[] constraint = constraints.get(con);

			for(int var = 0; var < variable_count; var++) {
				dictionary[con][var] = constraint[var];;
			}

			for(int slack = 0; slack < constraint_count; slack++) {
				dictionary[con][variable_count + slack] = con == slack ? 1 : 0;
			}

			dictionary[con][variable_count + constraint_count] = constraint[variable_count];
		}

		for(int var = 0; var < variable_count; var++) {
			dictionary[constraint_count][var] = minimize ? -objective[var] : objective[var];
		}
	}

	/**
	 * Selects a variable to enter the basis. If no viable candidates exist,
	 * returns -1.
	 */
	private int select_enter() {
		if(dictionary == null) {
			throw new RuntimeException("Attempted to select enter variable while not solving a problem");
		}

		double best = 0;
		int index = -1;
		for(int col = 0; col < dictionary_cols - 1; col++) {
			double curr = dictionary[dictionary_rows - 1][col];
			if(curr > best) {
				best = curr;
				index = col;
			}
		}

		return index;
	}

	/**
	 * Given the variable that will enter the basis, selects a row to exit the
	 * basis. Returns -1 if no viable candidates exist. 
	 */
	private int select_exit(int enter) {
		if(dictionary == null) {
			throw new RuntimeException("Attempted to select exit variable while not solving a problem");
		}
		if(enter < 0 || dictionary_cols - 1 <= enter) {
			throw new RuntimeException(String.format("Entering index %d is invalid for %d columns", enter, dictionary_cols));
		}

		double best = 0;
		int index = -1;
		for(int row = 0; row < dictionary_rows - 1; row++) {
			double check = dictionary[row][enter];
			if(index == -1 && check > 0) {
				best = check;
				index = row;
			}

			double curr = dictionary[row][dictionary_cols - 1] / check;
			if(curr < best) {
				best = curr;
				index = row;
			}
		}

		return index;
	}

	//////////////////////////////// OVERRIDING ////////////////////////////////

	@Override
	public void subject_to(double... params) {
		if(params.length != variable_count + 1) {
			throw new RuntimeException(String.format("%d values expected, %d given", variable_count + 1, params.length));
		}

		double[] constraint = new double[variable_count + 1];
		for(int var = 0; var < variable_count + 1; var++) {
			constraint[var] = params[var];
		}
		constraints.add(constraint);
		constraint_count++;
	}

	@Override
	public void maximize(double... params) {
		if(params.length != variable_count) {
			throw new RuntimeException(String.format("%d coefficients expected, %d given", variable_count, params.length));
		}

		objective = new double[variable_count];
		for(int var = 0; var < variable_count; var++) {
			objective[var] = params[var];
		}
	}

	@Override
	public void minimize(double... params) {
		if(params.length != variable_count) {
			throw new RuntimeException(String.format("%d coefficients expected, %d given", variable_count, params.length));
		}

		objective = new double[variable_count];
		for(int var = 0; var < variable_count; var++) {
			objective[var] = -params[var];
		}
	}

	@Override
	public boolean solve() {
		load_dictionary();
		infeasible = false;
		unbounded = false;

		int enter, exit;
		while(true) {
			enter = select_enter();
			if(enter == -1) break;

			exit = select_exit(enter);
			if(exit == -1) {
				unbounded = true;
				break;
			}

			pivot(enter, exit);
		}
		output = dictionary[dictionary_rows - 1][dictionary_cols - 1];
		params = new double[0][];

		dictionary = null;
		return false;
	}

	@Override
	public boolean is_infeasible() {
		if(!infeasible && !unbounded && params == null) {
			throw new RuntimeException("Solve problem before polling solution");
		}
		return infeasible;
	}

	@Override
	public boolean is_unbounded() {
		if(!infeasible && !unbounded && params == null) {
			throw new RuntimeException("Solve problem before polling solution");
		}
		return unbounded;
	}

	@Override
	public double value() {
		if(!infeasible && !unbounded && params == null) {
			throw new RuntimeException("Solve problem before polling solution");
		}
		if(infeasible || unbounded) {
			throw new RuntimeException("Attempted to poll infeasible/unbounded solution");
		}
		return minimize ? output : -output;
	}

	@Override
	public double[][] parameters() {
		if(!infeasible && !unbounded && params == null) {
			throw new RuntimeException("Solve problem before polling solution");
		}
		if(infeasible || unbounded) {
			throw new RuntimeException("Attempted to describe infeasible/unbounded solution");
		}

		double[][] out = new double[params.length][];
		for(int row = 0; row < out.length; row++) {
			out[row] = new double[ params[row].length ];
			for(int col = 0; col < out[row].length; col++) {
				out[row][col] = params[row][col];
			}
		}
		return out;
	}

}

