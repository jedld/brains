package com.dayosoft.nn;

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.ArrayList;

import com.dayosoft.nn.NeuralNet.Pair;
import com.dayosoft.nn.NeuralNet.Tuple;

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
	double biasWeight = 1f;
	private double previousWeight[];
	
	//for batch learning
	private double deltaWeight[];
	private double deltaBias;
	private double inputs[];
	private double output = 0.0f;

	private int id;
	private int layer;

	public int getId() {
		return id;
	}

	public void setId(int id) {
		this.id = id;
	}

	public Neuron(int id, int layer, int connections, double triggerValue, int activationFunctionType) {
		this.id = id;
		this.activationFunctionType = activationFunctionType;
		this.layer = layer;
		this.bias = triggerValue;
		this.totalConnections = connections;
		this.inputs = new double[connections];
		this.previousWeight = new double[connections];
		this.deltaWeight = new double[connections];
		this.setWeights(new double[connections]);
		this.deltaBias = 0.0f;
		for (int i = 0; i < connections; i++) {
			weight[i] = 0.01f;
			previousWeight[i] = 0.01f;
			deltaWeight[i] = 0.0f;
		}
	}
	
	public void resetDeltas() {
		this.deltaBias = 0.0f;
		for (int i = 0; i < totalConnections; i++) {
			deltaWeight[i] = 0.0f;
		}
	}

	public void reset() {
		output = 0.0f;
		for (int i = 0; i < totalConnections; i++) {
			inputs[i] = 0.0f;
		}
	}

	public void setInput(int id, double value) {
		inputs[id] = value;
	};

	public double adjustForOutput(double errorTerm, double learningRate, double momentum, boolean deltaOnly) {
		double sum = 0;
		int index = 0;
		double e = (1 - momentum) * learningRate * errorTerm;
		for (double t : inputs) {
			sum += errorTerm * weight[index];
			double pw = weight[index];
			double delta =  e * t
					+ momentum * (weight[index] - previousWeight[index]);
			if (!deltaOnly) {
				previousWeight[index] = pw;
				weight[index] += delta;
			} else {
				deltaWeight[index] += delta;
			}
			index++;
		}
		//update bias weight
		sum += errorTerm * this.biasWeight * this.bias ;
		double pBW = this.biasWeight;
		double deltaBias = e * this.bias + momentum * (biasWeight - previousBiasWeight);
		if (!deltaOnly) {
			this.biasWeight += deltaBias;
			this.previousBiasWeight = pBW;
		} else {
			this.deltaBias += deltaBias;
		}
		return sum;
	}
	
	public void applyDelta() {
		for (int i =0; i < this.totalConnections; i++) {
			previousWeight[i] = weight[i];
			weight[i] += deltaWeight[i];
			deltaWeight[i] = 0.0f;
		}
		this.previousBiasWeight = this.biasWeight;
		this.biasWeight += this.deltaBias;
		this.deltaBias = 0.0f;
	}

	public double derivative() {
		if (activationFunctionType == Neuron.SIGMOID) {
			return output * (1 - output);
		} else if (activationFunctionType == Neuron.HTAN){
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
		if (output == 0.0f) {
			if (activationFunctionType == SIGMOID) {
				output = (1f / (1f + Math.exp(-(getTotal()))));
			} else if (activationFunctionType == HTAN){
				output = Math.tanh( getTotal());
			} else if (activationFunctionType == SOFTMAX) {
				output = Math.log(1f + Math.exp(getTotal()));
			} else if (activationFunctionType == RECTIFIER) {
				output = getTotal() > 0f ? getTotal() : 0.1f * getTotal();
			}
			return output;
		} else {
			return output;
		}
	}

	public double getTotal() {
		double total = 0;
		int index = 0;
		for (double p : inputs) {
			total += p * weight[index++];
		}
		return total + biasWeight * bias;
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
}
