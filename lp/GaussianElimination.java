package apple_lib.lp;

/**
 * Reduces a matrix to reduced row echilon form using Gaussian elimination.
 */
public class GaussianElimination {

	////////////////////////////////// FIELDS //////////////////////////////////

	/* Base matrix */
	protected double[][] matrix;

	/* Sizes */
	protected int N, M;

	/////////////////////////////// CONSTRUCTORS ///////////////////////////////

	/**
	 * Basic constructor
	 */
	public GaussianElimination(int rows, int cols) {
		N = rows;
		M = cols;
		matrix = null;
	}

	////////////////////////////////// METHODS /////////////////////////////////

	/**
	 * Loads a matrix for solving. Ensures the matrix matches the fixed sizes. 
	 */
	public void load(double[][] mat) {
		if(mat.length != N) {
			throw new IllegalArgumentException();
		}
		double[][] out = new double[N][M];
		for(int row = 0; row < N; row++) {
			if(mat[row].length != M) {
				throw new IllegalArgumentException();
			}
			for(int col = 0; col < M; col++) {
				out[row][col] = mat[row][col];
			}
		}
		matrix = out;
	}

	/**
	 * Reduces loaded matrix to reduced row echilon form.
	 */
	public double[][] rref() {
		if(matrix == null) {
			throw new IllegalStateException();
		}

		// Initialize output variable
		double[][] out = new double[N][M];
		for(int row = 0; row < N; row++) {
			for(int col = 0; col < M; col++) {
				out[row][col] = matrix[row][col];
			}
		}
		int col_max = Math.min(N, M);

		// Store variable for each row
		int[] row_var = new int[N];
		for(int row = 0; row < N; row++) {
			row_var[row] = -1;
		}

		// Main loop
		for(int fill = 0; fill < N; fill++) {
			// Find largest coefficient
			double coeff = 0;
			int target_row = fill;
			int target_col = 0;
			for(int row = fill; row < N; row++) {
				for(int col = 0; col < col_max; col++) {
					double poll = Math.abs(out[row][col]);
					if(poll > coeff) {
						coeff = poll;
						target_row = row;
						target_col = col;
					}
				}
			}
			if(coeff == 0) {
				break;
			}

			// Bring fill row to top
			double[] swap = out[fill];
			out[fill] = out[target_row];
			out[target_row] = swap;
			row_var[fill] = target_col;

			// Normalize target row
			coeff = out[fill][target_col];
			for(int col = 0; col < M; col++) {
				out[fill][col] /= coeff;
			}

			// Zero other rows
			for(int row = fill + 1; row < N; row++) {
				double base = out[row][target_col];
				for(int col = 0; col < M; col++) {
					out[row][col] -= base * out[fill][col];
				}
			}
		}

		// Second pass
		double[][] reorder = new double[N][M];
		for(int row = N - 1; row >= 0; row--) {
			int target_col = row_var[row];
			if(target_col == -1) continue;

			for(int target = 0; target < row; target++) {
				double base = out[target][target_col];
				for(int col = 0; col < M; col++) {
					out[target][col] -= base * out[row][col];
				}
			}

			reorder[target_col] = out[row];
		}

		return reorder;
	}

}

