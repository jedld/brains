package com.dayosoft.nn;

public class NeuronInputs {
	protected double inputs[];
	protected double hiddenState = 0f;
	protected double output;
	protected int inputsize;
	
	public NeuronInputs(int inputsize) {
		this.inputsize = inputsize;
		this.inputs = new double[inputsize];
	}
}
