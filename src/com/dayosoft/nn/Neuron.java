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
	double previousBiasWeight;
	int activationFunctionType = SIGMOID;
	int totalConnections;
	private double previousWeight[];

	// for batch learning
	private double deltaWeight[];
	private double previousDeltaWeight[];
	private double weightUpdateValue[];
	private double weightChange[];
	protected double deltaBias = 0.0f;
	private double hiddenStateDeltaBias = 0.0f;
	private double biasUpdateValue = 0.0f;
	private double biasChange = 0.0f;

	public double getHiddenStateOutput() {
		return runtimeStates.hiddenState;
	}

	public void setHiddenStateOutput(double previousOutput) {
		runtimeStates.hiddenState = previousOutput;
	}

	public double getPreviousHiddenStateWeight() {
		return hiddenStateWeight;
	}

	public void setPreviousHiddenStateWeight(double previousHiddenStateWeight) {
		this.hiddenStateWeight = previousHiddenStateWeight;
	}

	private double hiddenStateWeight = 0.0f;
	private ArrayList<double[]> recordedInputs = new ArrayList<double[]>();

	protected boolean fired = false;
	protected double output = 0.0f;

	private int id;
	private int layer;
	
	private boolean isRecurrent = false;
	protected double previousHiddenStateOutputWeight;
	private double deltaPreviousHiddenStateOutputWeight;
	protected double previousErrorSumForNode;
	private NeuronInputs runtimeStates;
	protected ParameterSet parameterSet;

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
		this.runtimeStates = new NeuronInputs(connections);
		this.parameterSet = new ParameterSet(connections);
		setupNeuron(id, layer, connections, triggerValue, isRecurrent, activationFunctionType);		
	}
	
	public Neuron(NeuronInputs inputs, ParameterSet parameters, int id, int layer, int connections, double triggerValue, boolean isRecurrent, int activationFunctionType) {
		this.runtimeStates = inputs;
		this.parameterSet = parameters;
		setupNeuron(id, layer, connections, triggerValue, isRecurrent, activationFunctionType);
	}

	private void setupNeuron(int id, int layer, int connections, double triggerValue, boolean isRecurrent,
			int activationFunctionType) {
		this.id = id;
		this.activationFunctionType = activationFunctionType;
		this.layer = layer;
		this.parameterSet.bias = triggerValue;
		this.totalConnections = connections;
		this.previousWeight = new double[connections];
		this.deltaWeight = new double[connections];
		this.previousDeltaWeight = new double[connections];
		this.weightUpdateValue = new double[connections];
		this.weightChange = new double[connections];
		this.deltaBias = 0.0f;
		this.hiddenStateDeltaBias = 0.0f;
		this.biasChange = 0.0f;
		this.biasUpdateValue = 0.1f;
		this.hiddenStateWeight = 0.0f;
		this.isRecurrent  = isRecurrent;
		this.previousErrorSumForNode = 0.0f;
		
		for (int i = 0; i < connections; i++) {
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
		runtimeStates.inputs[id] = value;
	};

	public double computeForDelta(double errorTerm) {
		double sum = 0;
		for (int index = 0; index < runtimeStates.inputsize ; index++) {
			double delta = errorTerm * runtimeStates.inputs[index];
			sum += errorTerm * this.parameterSet.weights[index];
			deltaWeight[index] += delta;
			System.out.println("g " +deltaWeight[index]);
		}

		sum += errorTerm * this.parameterSet.biasWeight;
		this.deltaBias += errorTerm;
		System.out.println("g " +deltaBias);
		return sum;
	}

	public double adjustForOutput(double errorTerm, double learningRate, double momentum, boolean deltaOnly) {
		double errorSumForNode = 0;
		double e = (1 - momentum) * learningRate * errorTerm;
		for (int index = 0; index < runtimeStates.inputsize; index++) {
			errorSumForNode += errorTerm * this.parameterSet.weights[index];
			double pw = this.parameterSet.weights[index];
			double delta = e * runtimeStates.inputs[index] + momentum * (this.parameterSet.weights[index] - previousWeight[index]);
			if (deltaOnly) {
				deltaWeight[index] += delta;
			} else {
				previousWeight[index] = pw;
				this.parameterSet.weights[index] += delta;
			}
		}

		// update bias weight
		errorSumForNode += errorTerm * this.parameterSet.biasWeight * this.parameterSet.bias;
		double pBW = this.parameterSet.biasWeight;
		double deltaBias = e * this.parameterSet.bias + momentum * (this.parameterSet.biasWeight - previousBiasWeight);
		if (deltaOnly) {
			this.deltaBias += deltaBias;
		} else {
			this.parameterSet.biasWeight += deltaBias;
			this.previousBiasWeight = pBW;
		}
		
		// update previous output weight
		if (this.isRecurrent) {
			double hiddenStateError = errorTerm * this.hiddenStateWeight * runtimeStates.hiddenState;
			this.previousErrorSumForNode = hiddenStateError;
			errorSumForNode += hiddenStateError;
			double pOW = this.hiddenStateWeight;
			double deltaPreviousOutput = e * runtimeStates.hiddenState + momentum * (hiddenStateWeight - previousHiddenStateOutputWeight);
			if (deltaOnly) {
				this.deltaPreviousHiddenStateOutputWeight += deltaPreviousOutput;
			} else {
				this.hiddenStateWeight += deltaPreviousOutput;
				this.previousHiddenStateOutputWeight = pOW;
			}
		}

		return errorSumForNode;
	}

	public void applyDelta() {
		for (int i = 0; i < this.totalConnections; i++) {
			previousWeight[i] = this.parameterSet.weights[i];
			this.parameterSet.weights[i] += deltaWeight[i];
			deltaWeight[i] = 0.0f;
		}
		this.previousBiasWeight = this.parameterSet.biasWeight;
		this.parameterSet.biasWeight += this.deltaBias;
		this.deltaBias = 0.0f;
		
		this.previousHiddenStateOutputWeight = this.hiddenStateWeight;
		this.hiddenStateWeight += this.deltaPreviousHiddenStateOutputWeight;
		this.deltaPreviousHiddenStateOutputWeight = 0.0f;
	}

	public double incrementDelta(double value) {
		double sumDeltaWeights = 0;
		for (int i = 0; i < this.totalConnections; i++) {
			this.deltaWeight[i] += fire() * value;
			sumDeltaWeights += value * this.parameterSet.weights[i];
		}

		this.deltaBias += value;
		return sumDeltaWeights + (value * this.parameterSet.bias);
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
		for (int index = 0; index < runtimeStates.inputsize ; index++) {
			total += runtimeStates.inputs[index] * this.parameterSet.weights[index];
		}
		
		return total + runtimeStates.hiddenState * hiddenStateWeight + this.parameterSet.biasWeight * this.parameterSet.bias;
	}

	double[] getWeights() {
		return this.parameterSet.weights;
	}

	double[] getDeltas() {
		return deltaWeight;
	}

	void setWeights(double weights[]) {
		this.parameterSet.weights = weights;
		this.previousWeight = weights.clone();
	}

	public double getInput(int i) {
		return runtimeStates.inputs[i];
	}

	public void setWeights(int i2, double w) {
		this.parameterSet.weights[i2] = w;
		this.previousWeight[i2] = w;
	}

	public void setBiasWeight(double d) {
		this.parameterSet.biasWeight = d;
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

		applyRPropUpdates(-1, -this.deltaBias, this.hiddenStateDeltaBias, this.biasChange, this.biasUpdateValue);
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
			this.parameterSet.bias += weightChange;
			this.biasChange = weightChange;
			this.biasUpdateValue = weightUpdateValue;
			this.hiddenStateDeltaBias = gradient;
			System.out.println(this.parameterSet.bias);
		} else {
			this.parameterSet.weights[i] += weightChange;
			this.weightChange[i] = weightChange;
			this.weightUpdateValue[i] = weightUpdateValue;
			this.previousDeltaWeight[i] = gradient;
			System.out.println(this.parameterSet.weights[i] + " " + weightChange);
		}
	}

	public void resetRecurrenceStates() {
		runtimeStates.hiddenState = 0.0f;
		
		for(int i = 0; i < runtimeStates.inputsize; i++) {
			runtimeStates.inputs[i] = 0.0f;
		}
	}

	public void updatePreviousOutput() {
		if (fired) {
			runtimeStates.hiddenState = this.output;
		} else {
			//neuron not fired! This is not expected
			throw new RuntimeException();
		}
	}

	public double getDeltaPreviousOutput() {
		// TODO Auto-generated method stub
		return this.deltaPreviousHiddenStateOutputWeight;
	}

	public double[] getInputs() {
		return this.runtimeStates.inputs;
	}

	public void setInput(double[] newInput) {
		this.runtimeStates.inputs = newInput;
	}

	public void setOutput(double currentOutput) {
		this.output = currentOutput;
		this.fired = true;
	}
}
