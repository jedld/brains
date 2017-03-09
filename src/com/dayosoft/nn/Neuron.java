package com.dayosoft.nn;

import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.ArrayList;

public class Neuron {

	public static final int SIGMOID = 1;
	public static final int HTAN = 2;
	public static final int SOFTMAX = 3;
	public static final int RECTIFIER = 4;
	long seed;
	double bias;
	double previousBiasWeight;
	int activationFunctionType = SIGMOID;
	int totalConnections;
	private double weight[];
	double biasWeight = 0f;
	private double previousWeight[];

	// for batch learning
	private double deltaWeight[];
	private double previousDeltaWeight[];
	private double weightUpdateValue[];
	private double weightChange[];
	protected double deltaBias = 0.0f;
	private double previousDeltaBias = 0.0f;
	private double biasUpdateValue = 0.0f;
	private double biasChange = 0.0f;
	
	private double inputs[];
	
	//used for recurrent neural networks
	private double previousOutput = 0.0f;
	
	public double getPreviousOutput() {
		return previousOutput;
	}

	public void setPreviousOutput(double previousOutput) {
		this.previousOutput = previousOutput;
	}

	public double getPreviousOutputWeight() {
		return previousOutputWeight;
	}

	public void setPreviousOutputWeight(double previousOutputWeight) {
		this.previousOutputWeight = previousOutputWeight;
	}

	private double previousOutputWeight = 0.0f;
	private ArrayList<double[]> recordedInputs = new ArrayList<double[]>();

	protected boolean fired = false;
	protected double output = 0.0f;

	private int id;
	private int layer;
	
	private boolean isRecurrent = false;
	private double previousPreviousOutputWeight;
	private double deltaPreviousOutputWeight;

	public boolean isRecurrent() {
		return isRecurrent;
	}

	public void setRecurrent(boolean isRecurrent) {
		this.isRecurrent = isRecurrent;
	}

	public int getId() {
		return id;
	}

	public void setId(int id) {
		this.id = id;
	}

	public Neuron(int id, int layer, int connections, double triggerValue, boolean isRecurrent, int activationFunctionType) {
		this.id = id;
		this.activationFunctionType = activationFunctionType;
		this.layer = layer;
		this.bias = triggerValue;
		this.totalConnections = connections;
		this.inputs = new double[connections];
		this.previousWeight = new double[connections];
		this.deltaWeight = new double[connections];
		this.previousDeltaWeight = new double[connections];
		this.weightUpdateValue = new double[connections];
		this.weightChange = new double[connections];
		this.setWeights(new double[connections]);
		this.deltaBias = 0.0f;
		this.previousDeltaBias = 0.0f;
		this.biasChange = 0.0f;
		this.biasUpdateValue = 0.1f;
		this.previousOutput = 0.0f;
		this.previousOutputWeight = 0.0f;
		this.isRecurrent  = isRecurrent;
		
		for (int i = 0; i < connections; i++) {
			weight[i] = 0.00f;
			previousWeight[i] = 0.00f;
			deltaWeight[i] = 0.0f;
			previousDeltaWeight[i] = 0.0f;
			weightChange[i] = 0.0f;
			weightUpdateValue[i] = 0.1f;
		}
	}

	public void resetAllDeltas() {
		this.deltaBias = 0.0f;
		this.biasChange = 0.0f;
		this.biasUpdateValue = 0.1f;
		for (int i = 0; i < totalConnections; i++) {
			deltaWeight[i] = 0.0f;
			previousWeight[i] = 0.00f;
			deltaWeight[i] = 0.0f;
			previousDeltaWeight[i] = 0.0f;
			weightChange[i] = 0.0f;
			weightUpdateValue[i] = 0.1f;
		}
	}

	public void resetDeltas() {
		deltaBias = 0.0f;
		for (int i = 0; i < totalConnections; i++) {
			deltaWeight[i] = 0.0f;
		}
	}

	public void reset() {
		output = 0.0f;
		fired = false;
	}

	public void setInput(int id, double value) {
		inputs[id] = value;
	};

	public double computeForDelta(double errorTerm) {
		double sum = 0;
		for (int index = 0; index < inputs.length; index++) {
			double delta = errorTerm * inputs[index];
			sum += errorTerm * weight[index];
			deltaWeight[index] += delta;
			System.out.println("g " +deltaWeight[index]);
		}

		sum += errorTerm * this.biasWeight;
		this.deltaBias += errorTerm;
		System.out.println("g " +deltaBias);
		return sum;
	}

	public double adjustForOutput(double errorTerm, double learningRate, double momentum, boolean deltaOnly) {
		double sum = 0;
		double e = (1 - momentum) * learningRate * errorTerm;
		for (int index = 0; index < inputs.length; index++) {
			sum += errorTerm * weight[index];
			double pw = weight[index];
			double delta = e * inputs[index] + momentum * (weight[index] - previousWeight[index]);
			if (deltaOnly) {
				deltaWeight[index] += delta;
			} else {
				previousWeight[index] = pw;
				weight[index] += delta;
			}
		}

		// update bias weight
		sum += errorTerm * this.biasWeight * this.bias;
		double pBW = this.biasWeight;
		double deltaBias = e * this.bias + momentum * (biasWeight - previousBiasWeight);
		if (deltaOnly) {
			this.deltaBias += deltaBias;
		} else {
			this.biasWeight += deltaBias;
			this.previousBiasWeight = pBW;
		}
		
		// update previous output weight
		sum += errorTerm * this.previousOutputWeight * this.previousOutput;
		double pOW = this.previousOutputWeight;
		double deltaPreviousOutput = e * this.previousOutput + momentum * (previousOutputWeight - previousPreviousOutputWeight);
		if (deltaOnly) {
			this.deltaPreviousOutputWeight += deltaPreviousOutput;
		} else {
			this.previousOutputWeight += deltaPreviousOutput;
			this.previousPreviousOutputWeight = pOW;
		}
		return sum;
	}

	public void applyDelta() {
		for (int i = 0; i < this.totalConnections; i++) {
			previousWeight[i] = weight[i];
			weight[i] += deltaWeight[i];
			deltaWeight[i] = 0.0f;
		}
		this.previousBiasWeight = this.biasWeight;
		this.biasWeight += this.deltaBias;
		this.deltaBias = 0.0f;
		
		this.previousPreviousOutputWeight = this.previousOutputWeight;
		this.previousOutputWeight += this.deltaPreviousOutputWeight;
		this.deltaPreviousOutputWeight = 0.0f;
	}

	public double incrementDelta(double value) {
		double sumDeltaWeights = 0;
		for (int i = 0; i < this.totalConnections; i++) {
			this.deltaWeight[i] += fire() * value;
			sumDeltaWeights += value * weight[i];
		}

		this.deltaBias += value;
		return sumDeltaWeights + (value * bias);
	}

	public double derivative() {
		if (activationFunctionType == Neuron.SIGMOID) {
			return output * (1 - output);
		} else if (activationFunctionType == Neuron.HTAN) {
			return 1 - Math.pow(output, 2);
		} else if (activationFunctionType == SOFTMAX || activationFunctionType == RECTIFIER) {
			return (1f / (1f + Math.exp(-output)));
		}
		return 0;
	}

	String round(double val) {
		DecimalFormat df = new DecimalFormat("#.#######");
		df.setRoundingMode(RoundingMode.CEILING);
		return df.format(val);
	}

	public double fire() {
		if (!fired) {
			if (activationFunctionType == SIGMOID) {
				output = (1f / (1f + Math.exp(-(getTotal()))));
			} else if (activationFunctionType == HTAN) {
				output = Math.tanh(getTotal());
			} else if (activationFunctionType == SOFTMAX) {
				output = Math.log(1f + Math.exp(getTotal()));
			} else if (activationFunctionType == RECTIFIER) {
				output = getTotal() > 0f ? getTotal() : 0.1f * getTotal();
			}
			fired = true;
			return output;
		} else {
			return output;
		}
	}

	public double getTotal() {
		double total = 0;
		for (int index = 0; index < inputs.length; index++) {
			total += inputs[index] * weight[index];
		}
		
		return total + previousOutput * previousOutputWeight + biasWeight * bias;
	}

	double[] getWeights() {
		return weight;
	}

	double[] getDeltas() {
		return deltaWeight;
	}

	void setWeights(double weights[]) {
		this.weight = weights;
		this.previousWeight = weights.clone();
	}

	public double getInput(int i) {
		return inputs[i];
	}

	public void setWeights(int i2, double w) {
		this.weight[i2] = w;
		this.previousWeight[i2] = w;
	}

	public void setBiasWeight(double d) {
		this.biasWeight = d;
		this.previousBiasWeight = d;
	}

	public static final double TOLERANCE = Math.exp(-16);

	private int sign(double s) {
		if (s > TOLERANCE)
			return 1;
		if (s < -TOLERANCE)
			return -1;
		return 0;
	}

	final double MIN_STEP = Math.exp(-6), MAX_STEP = 50f;

	public void updateGradient() {
		for (int i = 0; i < this.totalConnections; i++) {
			double gradient = -this.deltaWeight[i];
			double previousGradient = this.previousDeltaWeight[i];
			double weightChange = this.weightChange[i];
			double weightUpdateValue = this.weightUpdateValue[i];
//
//			System.out.println(
//					i + "-> " + gradient + " " + previousGradient + " " + weightChange + " " + weightUpdateValue);
			applyRPropUpdates(i, gradient, previousGradient, weightChange, weightUpdateValue);
		}

		applyRPropUpdates(-1, -this.deltaBias, this.previousDeltaBias, this.biasChange, this.biasUpdateValue);
	}

	private void applyRPropUpdates(int i, double gradient, double previousGradient, double weightChange,
			double weightUpdateValue) {
		int c = sign(gradient * previousGradient);

		switch (c) {
		case 1:
			weightUpdateValue = Math.min(weightUpdateValue * 1.2, MAX_STEP);
			weightChange = ((double) -sign(gradient)) * weightUpdateValue;
			break;
		case -1:
			weightUpdateValue = Math.max(weightUpdateValue * 0.5, MIN_STEP);
			weightChange = -weightChange; // roll back previous weight change
			gradient = 0; // so won't trigger sign change on next update
			break;
		case 0:
			weightChange = ((double) -sign(gradient)) * weightUpdateValue;
		}

		if (i == -1) {
			this.bias += weightChange;
			this.biasChange = weightChange;
			this.biasUpdateValue = weightUpdateValue;
			this.previousDeltaBias = gradient;
			System.out.println(this.bias);
		} else {
			this.weight[i] += weightChange;
			this.weightChange[i] = weightChange;
			this.weightUpdateValue[i] = weightUpdateValue;
			this.previousDeltaWeight[i] = gradient;
			System.out.println(this.weight[i] + " " + weightChange);
		}
	}

	public void resetRecurrenceStates() {
		this.previousOutput = 0.0f;
	}

	public void setRecordedInput(int index) {
		this.inputs = this.recordedInputs.get(index);
	}

	public void updatePreviousOutput() {
		if (fired) {
			this.previousOutput = this.output;
		} else {
			//neuron not fired! This is not expected
			throw new RuntimeException();
		}
	}

	public double getDeltaPreviousOutput() {
		// TODO Auto-generated method stub
		return this.deltaPreviousOutputWeight;
	}

	public double[] getInputs() {
		return this.inputs;
	}
}
