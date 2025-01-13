package apple_lib.utility;

public interface Parameterizable {

	/**
	 * Encodes this object into a double array
	 */
	public double[] parameterize();

	/**
	 * Returns the length of the parameter representation
	 */
	public int parameter_count();

}

