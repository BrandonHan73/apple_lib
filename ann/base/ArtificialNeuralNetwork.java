package apple_lib.ann.base;

public interface ArtificialNeuralNetwork {

	public void load_input(double... input);
	public double[] test_input(double... input);

	public double[] calculate(int input);
	public double[] calculate();

	public void clear_inputs();

	public void load_derivative(int input, double... dCdy);
	public void load_derivative(double... dCdy);

}

