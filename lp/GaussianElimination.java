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
	 * Reduces loaded matrix to row echilon form.
	 */
	public double[][] ref() {
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

		// Main loop
		for(int fill = 0; fill < N; fill++) {

			// Find largest coefficient
			double coeff = 0;
			int target_row = fill;
			int target_col = 0;
			for(int row = 0; row < N; row++) {
				for(int col = 0; col < M; col++) {
					if(out[row][col] > coeff) {
						coeff = Math.abs(out[row][col]);
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

			// Normalize target row
			coeff = out[fill][target_col];
			for(int col = 0; col < M; col++) {
			}

		}

		return out;
	}

}

