package com.dayosoft.nn;

public class ParameterSet {
	protected double bias;
	protected double biasWeight;
	protected double hiddenStateWeight;
	protected double weights[];
	protected int inputsize;
	
	public ParameterSet(int inputsize) {
		this.inputsize = inputsize;
		weights = new double[inputsize];
		for(int i =0; i < inputsize; i++) {
			weights[i] = 0.00f;
		}
	}
}
