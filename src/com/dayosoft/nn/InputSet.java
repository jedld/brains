package com.dayosoft.nn;

import java.util.ArrayList;
import java.util.List;

public class InputSet {
	ArrayList<double[]> inputsList = new ArrayList<double[]>();
	ArrayList<Double> previousOutputList = new ArrayList<Double>();
	ArrayList<Double> currentOutputList = new ArrayList<Double>();
	
	public void addState(List<Neuron> neurons) {
		for(Neuron n : neurons) {
			inputsList.add(n.getInputs().clone());
			previousOutputList.add(n.getHiddenStateOutput());
			currentOutputList.add(n.output);
		}
	}
	
	public void restoreState(List<Neuron> neurons) {
		int index = 0;
		for(Neuron n : neurons) {
			double[] inputs = inputsList.get(index);
			double previousOutput = previousOutputList.get(index);
			double currentOutput = currentOutputList.get(index);
			n.setInput(inputs);
			n.setHiddenStateOutput(previousOutput);
			n.setOutput(currentOutput);
			
			index++;
		}
	}

}
