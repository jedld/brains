package com.dayosoft.nn;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import com.dayosoft.nn.utils.OutputUtils;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonIOException;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.JsonSyntaxException;

public class NeuralNet {
	private int neurons = 0;
	private int surface = 0;
	private int outputCount;
	ArrayList<ArrayList<Neuron>> layers = new ArrayList<>();
	ArrayList<Neuron> allNeurons = new ArrayList<Neuron>();
	private int layerCount;
	private int hiddenCount;
	private double learningRate = 1f;
	private Config config;
	private int neuronsPerLayer;
	private double bias;
	private double outputBias;
	private double momentumFactor;
	private int activationFunctionType;
	private int outputActivationFunctionType;
	private int errorFormula;
	private int gradientFormula;
	private int maxNeuronWidth;
	private boolean isRecurrent = false;

	public NeuralNet loadStateFromJson(String jsonFile) {
		try {
			return NeuralNet.loadStateFromJson(this, jsonFile);
		} catch (JsonIOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (JsonSyntaxException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}

	public static NeuralNet loadStateFromJsonString(NeuralNet nn, String jsonString)
			throws JsonIOException, JsonSyntaxException, FileNotFoundException {
		JsonParser parser = new JsonParser();
		JsonElement output = parser.parse(jsonString);

		return parseState(nn, output);
	}

	public static NeuralNet loadStateFromJson(NeuralNet nn, String jsonFile)
			throws JsonIOException, JsonSyntaxException, FileNotFoundException {
		JsonParser parser = new JsonParser();
		JsonElement output = parser.parse(new FileReader(jsonFile));

		return parseState(nn, output);
	}

	private static NeuralNet parseState(NeuralNet nn, JsonElement output) {
		JsonObject rootElement = output.getAsJsonObject();
		JsonArray layers = rootElement.get("neurons").getAsJsonArray();
		int inputCount = rootElement.get("surface").getAsInt();
		int outputCount = rootElement.get("outputCount").getAsInt();

		Config config = new Config(inputCount, outputCount, layers.size());
		config.neuronCount = rootElement.get("n").getAsInt();
		config.learningRate = rootElement.get("learningRate").getAsDouble();
		config.momentumFactor = rootElement.get("momentum").getAsDouble();
		config.neuronsPerLayer = rootElement.get("neuronsPerLayer").getAsInt();
		config.activationFunctionType = rootElement.get("activationFunction").getAsInt();
		config.outputActivationFunctionType = rootElement.get("outputActivationFunctionType").getAsInt();
		config.bias = rootElement.get("bias").getAsDouble();
		config.gradientFormula = rootElement.get("gradientFormula").getAsInt();
		config.outputBias = rootElement.get("outputBias").getAsDouble();
		config.gradientFormula = rootElement.get("gradientFormula").getAsInt();
		config.isRecurrent = rootElement.get("isRecurrent") != null ? rootElement.get("isRecurrent").getAsBoolean()
				: false;

		ArrayList<ArrayList<double[]>> weights = new ArrayList<ArrayList<double[]>>();
		ArrayList<ArrayList<Double>> biasWeights = new ArrayList<ArrayList<Double>>();
		ArrayList<ArrayList<Double>> hiddenStateWeights = new ArrayList<ArrayList<Double>>();

		for (int i = 0; i < layers.size(); i++) {
			JsonArray layer = layers.get(i).getAsJsonArray();
			ArrayList<double[]> n = new ArrayList<double[]>();
			ArrayList<Double> layerBiases = new ArrayList<Double>();
			ArrayList<Double> layerHiddenStateWeights = new ArrayList<Double>();
			for (int i2 = 0; i2 < layer.size(); i2++) {
				JsonObject neuronObj = layer.get(i2).getAsJsonObject();
				JsonArray weightArr = neuronObj.getAsJsonArray("weights");
				layerBiases.add(neuronObj.get("biasWeight").getAsDouble());
				if (config.isRecurrent) {
					layerHiddenStateWeights.add(neuronObj.get("previousOutputWeight").getAsDouble());
				}
				double w[] = new double[weightArr.size()];
				for (int i3 = 0; i3 < weightArr.size(); i3++) {
					w[i3] = weightArr.get(i3).getAsDouble();
				}
				n.add(w);
			}
			biasWeights.add(layerBiases);
			if (config.isRecurrent) {
				hiddenStateWeights.add(layerHiddenStateWeights);
			}
			weights.add(n);
		}

		if (nn == null) {
			nn = new NeuralNet(config);
		}

		nn.loadWeights(weights);
		nn.loadBiases(biasWeights);
		if (config.isRecurrent) {
			nn.loadHiddenStateWeights(hiddenStateWeights);
		}
		return nn;
	}

	private void loadHiddenStateWeights(ArrayList<ArrayList<Double>> hiddenStateWeights) {
		int i = 0;
		for (ArrayList<Double> n : hiddenStateWeights) {
			ArrayList<Neuron> layer = layers.get(i++);
			int i2 = 0;
			for (double w : n) {
				Neuron neuron = layer.get(i2++);
				neuron.setPreviousOutputWeight(w);
			}
		}
	}

	public String saveStateToJson() {
		return saveStateToJson(false);
	}
	
	public String saveStateToJson(boolean withInputs) {
		JsonObject jsonObject = new JsonObject();
		jsonObject.addProperty("n", neurons);
		jsonObject.addProperty("surface", surface);
		jsonObject.addProperty("momentum", new BigDecimal(momentumFactor));
		jsonObject.addProperty("neuronsPerLayer", neuronsPerLayer);
		jsonObject.addProperty("outputCount", outputCount);
		jsonObject.addProperty("learningRate", new BigDecimal(learningRate));
		jsonObject.addProperty("bias", new BigDecimal(bias));
		jsonObject.addProperty("gradientFormula", gradientFormula);
		jsonObject.addProperty("outputBias", new BigDecimal(outputBias));
		jsonObject.addProperty("activationFunction", activationFunctionType);
		jsonObject.addProperty("outputActivationFunctionType", outputActivationFunctionType);
		jsonObject.addProperty("isRecurrent", isRecurrent);

		JsonArray layerArray = new JsonArray();
		jsonObject.add("neurons", layerArray);

		for (ArrayList<Neuron> layer : layers) {
			JsonArray neuronLayer = new JsonArray();
			for (Neuron n : layer) {
				JsonObject nObj = new JsonObject();
				JsonArray w = new JsonArray();
				JsonArray inputs = new JsonArray();
				for (double d : n.getWeights()) {
					w.add(d);
				}
				for (double d : n.getInputs()) {
					inputs.add(d);
				}

				nObj.addProperty("bias", n.bias);
				nObj.addProperty("biasWeight", n.biasWeight);
				nObj.add("weights", w);
				nObj.addProperty("previousOutputWeight", n.getPreviousOutputWeight());
				if (withInputs) {
					nObj.add("inputs", inputs);
					nObj.addProperty("previous_output", n.getPreviousOutput());
					nObj.addProperty("output", n.output);
				}

				neuronLayer.add(nObj);
			}
			layerArray.add(neuronLayer);
		}

		return jsonObject.toString();
	}

	public static class Pair<T1, T2> {
		T1 p1;
		T2 p2;

		public Pair(T1 p1, T2 p2) {
			this.p1 = p1;
			this.p2 = p2;
		}

		public T1 first() {
			return p1;
		}

		public T2 second() {
			return p2;
		}
	};

	public static class Tuple<T0, T1, T2> {
		T0 p0;
		T1 p1;
		T2 p2;

		public Tuple(T0 p0, T1 p1, T2 p2) {
			this.p0 = p0;
			this.p1 = p1;
			this.p2 = p2;
		}

		public T0 first() {
			return p0;
		}

		public T1 second() {
			return p1;
		}

		public T2 third() {
			return p2;
		}
	};

	public static class Config {
		public static final int MEAN_SQUARED = 1;
		public static final int CROSS_ENTROPY = 2;
		public static int DERIVATIVE = 1;
		public static int CE = 2;
		public static final int STANDARD_BACKPROPAGATION = 1;
		public static final int RPROP_BACKPROPAGATION = 2;
		public double momentumFactor;
		public int neuronCount, inputCount, outputCount;
		public int neuronsPerLayer;
		public double learningRate;
		public double bias;
		public double outputBias;
		public int activationFunctionType;
		public int outputActivationFunctionType;
		public int errorFormula;
		public int gradientFormula;
		public int backPropagationAlgorithm;
		public boolean isRecurrent = false;

		public Config(int inputCount, int outputCount, int neuronCount) {
			this.inputCount = inputCount;
			this.neuronCount = neuronCount;
			this.outputCount = outputCount;
			this.bias = 1f;
			this.outputBias = 0f;
			this.learningRate = 0.02f;
			this.neuronsPerLayer = 3;
			this.momentumFactor = 0f;
			this.activationFunctionType = Neuron.SIGMOID;
			this.outputActivationFunctionType = Neuron.SIGMOID;
			this.errorFormula = Config.MEAN_SQUARED;
			this.gradientFormula = Config.DERIVATIVE;
			this.backPropagationAlgorithm = Config.STANDARD_BACKPROPAGATION;
			this.isRecurrent = false;
		}
	}

	public ArrayList<ArrayList<double[]>> dumpStates(boolean delta) {
		if (delta) {
			System.out.println("---------------- DELTA ONLY ---------------------");
		}
		System.out.println("learningRate = " + this.learningRate);
		ArrayList<ArrayList<double[]>> list = new ArrayList<ArrayList<double[]>>();
		for (int i = 0; i < layerCount; i++) {
			System.out.println("============= " + i + " ===============");
			ArrayList<Neuron> layer = layers.get(i);
			ArrayList<double[]> doubleArr = new ArrayList<double[]>();
			for (Neuron n : layer) {
				System.out.print("n:" + n.getId() + " -> ");
				if (delta) {
					doubleArr.add(n.getDeltas().clone());
				} else {
					doubleArr.add(n.getWeights().clone());
				}
				int index = 0;
				for (double f : n.getWeights()) {
					System.out.print(f + "(" + n.getInput(index++) + ") ");
				}
				System.out.println(
						" (p:" + n.getPreviousOutput() + ") (b:" + n.bias + ") " + n.getTotal() + " = " + n.fire());
			}
			list.add(doubleArr);
		}
		return list;
	}

	public ArrayList<ArrayList<double[]>> dumpWeights() {
		ArrayList<ArrayList<double[]>> list = new ArrayList<ArrayList<double[]>>();
		for (int i = 0; i < layerCount; i++) {
			ArrayList<Neuron> layer = layers.get(i);
			ArrayList<double[]> doubleArr = new ArrayList<double[]>();
			for (Neuron n : layer) {
				doubleArr.add(n.getWeights().clone());
			}
			list.add(doubleArr);
		}
		return list;
	}

	public ArrayList<ArrayList<Double>> dumpBiases() {
		ArrayList<ArrayList<Double>> biasLayer = new ArrayList<ArrayList<Double>>();
		for (int i = 0; i < layerCount; i++) {
			ArrayList<Neuron> layer = layers.get(i);
			ArrayList<Double> doubleArr = new ArrayList<Double>();
			for (Neuron n : layer) {
				doubleArr.add(n.bias);
			}
			biasLayer.add(doubleArr);
		}
		return biasLayer;
	}

	public ArrayList<ArrayList<Double>> dumpPreviousOutputs() {
		ArrayList<ArrayList<Double>> biasLayer = new ArrayList<ArrayList<Double>>();
		for (int i = 0; i < layerCount; i++) {
			ArrayList<Neuron> layer = layers.get(i);
			ArrayList<Double> doubleArr = new ArrayList<Double>();
			for (Neuron n : layer) {
				doubleArr.add(n.getPreviousOutput());
			}
			biasLayer.add(doubleArr);
		}
		return biasLayer;
	}

	public ArrayList<double[]> dumpDeltaPreviousOutput() {
		ArrayList<double[]> list = new ArrayList<double[]>();

		for (int i = 0; i < layerCount; i++) {
			ArrayList<Neuron> layer = layers.get(i);
			double[] doubleArr = new double[layerCount];
			int index = 0;
			for (Neuron n : layer) {
				doubleArr[index++] = n.getDeltaPreviousOutput();
			}
			list.add(doubleArr);
		}
		return list;
	}

	public ArrayList<ArrayList<double[]>> dumpDeltas() {
		ArrayList<ArrayList<double[]>> list = new ArrayList<ArrayList<double[]>>();
		for (int i = 0; i < layerCount; i++) {
			ArrayList<Neuron> layer = layers.get(i);
			ArrayList<double[]> doubleArr = new ArrayList<double[]>();
			for (Neuron n : layer) {
				doubleArr.add(n.getDeltas().clone());
			}
			list.add(doubleArr);
		}
		return list;
	}

	public ArrayList<ArrayList<Double>> dumpDeltaBiases() {
		ArrayList<ArrayList<Double>> list = new ArrayList<ArrayList<Double>>();
		for (int i = 0; i < layerCount; i++) {
			ArrayList<Neuron> layer = layers.get(i);
			ArrayList<Double> doubleArr = new ArrayList<Double>();
			for (Neuron n : layer) {
				doubleArr.add(n.deltaBias);
			}
			list.add(doubleArr);
		}
		return list;
	}

	public int getNeurons() {
		return neurons;
	}

	public int getSurface() {
		return surface;
	}

	public int getOutputCount() {
		return outputCount;
	}

	public ArrayList<ArrayList<Neuron>> getLayers() {
		return layers;
	}

	public ArrayList<Neuron> getAllNeurons() {
		return allNeurons;
	}

	public int getLayerCount() {
		return layerCount;
	}

	public int getHiddenCount() {
		return hiddenCount;
	}

	public double getLearningRate() {
		return learningRate;
	}

	public Config getConfig() {
		return config;
	}

	public int getNeuronsPerLayer() {
		return neuronsPerLayer;
	}

	public double getBias() {
		return bias;
	}

	public double getOutputBias() {
		return outputBias;
	}

	public double getMomentumFactor() {
		return momentumFactor;
	}

	public int getActivationFunctionType() {
		return activationFunctionType;
	}

	public int getOutputActivationFunctionType() {
		return outputActivationFunctionType;
	}

	public int getErrorFormula() {
		return errorFormula;
	}

	public int getGradientFormula() {
		return gradientFormula;
	}

	public void loadWeights(ArrayList<ArrayList<double[]>> saveList) {
		int i = 0;
		for (ArrayList<double[]> n : saveList) {
			ArrayList<Neuron> layer = layers.get(i++);
			int i2 = 0;
			for (double[] w : n) {
				Neuron neuron = layer.get(i2++);
				neuron.setWeights(w);
			}
		}
	}

	public void loadBiases(ArrayList<ArrayList<Double>> saveList) {
		int i = 0;
		for (ArrayList<Double> n : saveList) {
			ArrayList<Neuron> layer = layers.get(i++);
			int i2 = 0;
			for (double w : n) {
				Neuron neuron = layer.get(i2++);
				neuron.setBiasWeight(w);
			}
		}
	}

	public void randomizeWeights(double min, double max) {
		Random random = new Random(System.currentTimeMillis());
		double range = max - min;
		for (int i = 0; i < layerCount; i++) {
			ArrayList<Neuron> layer = layers.get(i);
			for (Neuron n : layer) {
				double[] weights = n.getWeights();
				for (int i2 = 0; i2 < weights.length; i2++) {
					n.setWeights(i2, range * random.nextDouble() + min);
				}
				n.setBiasWeight(range * random.nextDouble() + min);
				if (this.config.isRecurrent) {
					n.setPreviousOutputWeight(range * random.nextDouble() + min);
				}
			}
		}
	}

	public void updatePreviousOutputs() {
		for (Neuron n : this.allNeurons) {
			n.updatePreviousOutput();
		}
	}

	public void computeGradients2(double expectedOutput[]) {
		int i2 = 0;
		double previousLayerDelta = 0;
		for (Neuron n : layers.get(layerCount - 1)) {
			double delta = n.derivative() * (expectedOutput[i2++] - n.fire());
			previousLayerDelta += n.incrementDelta(delta);
		}

		for (int i = layerCount - 2; i >= 0; i--) {
			ArrayList<Neuron> layer = layers.get(i);
			double error = previousLayerDelta;
			previousLayerDelta = 0;
			for (Neuron n : layer) {
				double delta = n.derivative() * error;
				previousLayerDelta += n.incrementDelta(delta);
			}
		}
	}

	public void computeGradients(double expectedOutput[]) {
		double deltaSum = 0;
		// System.out.println("========== training ============");
		int i2 = 0;
		for (Neuron n : layers.get(layerCount - 1)) {
			double errorTerm = n.derivative() * -(n.fire() - expectedOutput[i2++]);
			deltaSum += n.computeForDelta(errorTerm);
		}

		for (int i = layerCount - 2; i >= 0; i--) {
			// System.out.println("----------- Layer " + i + " --------------");
			ArrayList<Neuron> layer = layers.get(i);
			double currentDeltaSum = 0;
			for (Neuron n : layer) {
				double errorTerm = n.derivative() * deltaSum;
				currentDeltaSum += n.computeForDelta(errorTerm);
			}
			deltaSum = currentDeltaSum;
		}
	}

	public void adjustWeights(double expectedOutput[]) {
		adjustWeights(expectedOutput, false);
	}

	public void adjustWeights(double expectedOutput[], boolean recordDelta) {
		double deltaSum = 0;
		// System.out.println("========== training ============");
		int i2 = 0;
		for (Neuron n : layers.get(layerCount - 1)) {
			double errorTerm;

			if (this.gradientFormula == Config.DERIVATIVE) {
				errorTerm = n.derivative() * (expectedOutput[i2++] - n.fire() + n.previousErrorSumForNode);
			} else {
				errorTerm = expectedOutput[i2++] - n.fire() + n.previousErrorSumForNode;
			}
			double nodeError = n.adjustForOutput(errorTerm, learningRate, momentumFactor, recordDelta);
			if (n.isRecurrent())
				n.previousErrorSumForNode = nodeError;
			deltaSum += nodeError;
		}

		for (int i = layerCount - 2; i >= 0; i--) {
			ArrayList<Neuron> layer = layers.get(i);
			double currentDeltaSum = 0;
			for (Neuron n : layer) {
				double errorTerm = n.derivative() * (deltaSum + n.previousErrorSumForNode);
				double nodeError = n.adjustForOutput(errorTerm, learningRate, momentumFactor, recordDelta);
				if (n.isRecurrent())
					n.previousErrorSumForNode = nodeError;
				currentDeltaSum += nodeError;
			}
			deltaSum = currentDeltaSum;
		}
	}

	public NeuralNet(Config config) {
		this.config = config;
		this.neurons = config.neuronCount;
		this.surface = config.inputCount;
		this.outputCount = config.outputCount;
		this.learningRate = config.learningRate;
		this.neuronsPerLayer = config.neuronsPerLayer;
		this.layerCount = neurons / config.neuronsPerLayer;
		this.bias = config.bias;
		this.momentumFactor = config.momentumFactor;
		this.outputBias = config.outputBias;
		this.activationFunctionType = config.activationFunctionType;
		this.outputActivationFunctionType = config.outputActivationFunctionType;
		this.errorFormula = config.errorFormula;
		this.gradientFormula = config.gradientFormula;
		this.maxNeuronWidth = surface;
		this.isRecurrent = config.isRecurrent;

		if (config.neuronsPerLayer > maxNeuronWidth) {
			maxNeuronWidth = neuronsPerLayer;
		}

		if (config.outputCount > maxNeuronWidth) {
			maxNeuronWidth = outputCount;
		}

		if (neurons % config.neuronsPerLayer != 0) {
			this.layerCount += 1;
		}

		// System.out.println("creating " + this.layerCount + " layers ");
		// System.out.println("Learning Rate " + this.learningRate);

		// if (this.activationFunctionType == Neuron.HTAN) {
		// System.out.println("Using Hyperbolic Tan()");
		// } else {
		// System.out.println("Using Symboid()");
		// }
		for (int i = 0; i < layerCount; i++) {
			ArrayList<Neuron> aLayer = new ArrayList<>();
			Neuron neuron;
			if (i == 0) {
				for (int i2 = 0; i2 < config.neuronsPerLayer; i2++) {
					neuron = new Neuron(i2, i, config.inputCount, config.bias, config.isRecurrent,
							this.activationFunctionType);
					aLayer.add(neuron);
					allNeurons.add(neuron);
				}
			} else if (i == layerCount - 1) {
				for (int i2 = 0; i2 < outputCount; i2++) {
					neuron = new Neuron(i2, i, config.neuronsPerLayer, config.outputBias, config.isRecurrent,
							this.outputActivationFunctionType);
					aLayer.add(neuron);
					allNeurons.add(neuron);
				}
			} else {
				for (int i2 = 0; i2 < config.neuronsPerLayer; i2++) {
					neuron = new Neuron(i2, i, config.neuronsPerLayer, config.bias, config.isRecurrent,
							this.activationFunctionType);
					aLayer.add(neuron);
					allNeurons.add(neuron);
				}
			}

			layers.add(aLayer);
		}

	}

	public void resetRecurrentStates() {
		for (Neuron n : this.allNeurons) {
			n.resetRecurrenceStates();
		}
	}

	public ArrayList<double[]> feed(ArrayList<double[]> lines) {
		ArrayList<double[]> results = new ArrayList<double[]>();
		resetRecurrentStates();
		double[] result = feed(lines.get(0));
		results.add(result);
		updatePreviousOutputs();
		for (int i = 1; i < lines.size(); i++) {
			result = feed(lines.get(i));
			updatePreviousOutputs();
			results.add(result);
		}

		return results;
	}

	/**
	 * Evaluate the neural net function
	 * 
	 * @param inputs
	 * @param reset
	 * @return
	 */
	public double[] feed(double inputs[]) {
		double output[] = new double[outputCount];
		for (Neuron n : allNeurons) {
			n.fired = false;
		}

		double layerOutputs[] = new double[maxNeuronWidth];

		ArrayList<Neuron> thisLayer = layers.get(0);
		int currentFireListSize = thisLayer.size();

		for (int index = 0; index < inputs.length; index++) {
			for (Neuron neuron : thisLayer) {
				neuron.setInput(index, inputs[index]);
			}
		}

		fireLayer(layerOutputs, thisLayer);

		for (int currentLayer = 1; currentLayer < layerCount - 1; currentLayer++) {
			thisLayer = layers.get(currentLayer);
			setInputs(layerOutputs, currentFireListSize, thisLayer);
			currentFireListSize = thisLayer.size();
			fireLayer(layerOutputs, thisLayer);
		}

		thisLayer = layers.get(layerCount - 1);
		setInputs(layerOutputs, currentFireListSize, thisLayer);
		fireLayer(output, thisLayer);

		return output;
	}

	private void fireLayer(double[] fireList, ArrayList<Neuron> thisLayer) {
		for (int index = 0; index < thisLayer.size(); index++) {
			Neuron neuron = thisLayer.get(index);
			fireList[index] = neuron.fired ? neuron.output : neuron.fire();
		}
	}

	private void setInputs(double[] currentFireList, int size, ArrayList<Neuron> thisLayer) {
		for (int index = 0; index < size; index++) {
			for (Neuron neuron : thisLayer) {
				neuron.setInput(index, currentFireList[index]);
			}
		}
	}

	public double computeAcccuracy(double output[], double expectedOutput[]) {
		double difference = 0;
		if (this.errorFormula == Config.MEAN_SQUARED) {
			for (int i = 0; i < output.length; i++) {
				difference += expectedOutput[i] - output[i];
			}
			return Math.pow(difference, 2);
		} else if (this.errorFormula == Config.CROSS_ENTROPY) {
			for (int i = 0; i < output.length; i++) {
				difference += Math.log(output[i]) * expectedOutput[i];
			}
		}
		return difference;
	}

	// Implementing Fisher–Yates shuffle
	void shuffleArray(Random rnd, ArrayList input, ArrayList output) {
		for (int i = input.size() - 1; i > 0; i--) {
			int index = rnd.nextInt(i + 1);
			// Simple swap
			swapElement(input, i, index);
			swapElement(output, i, index);
		}
	}

	private void swapElement(ArrayList<Object> input, int i, int index) {
		Object a = input.get(index);
		input.set(index, input.get(i));
		input.set(i, a);
	}

	public Pair<Integer, Double> optimize(ArrayList<double[]> in, ArrayList<double[]> exp, double target, int maxEpochs,
			int callBackInterval, OptimizationListener listener) {
		@SuppressWarnings("unchecked")
		ArrayList<double[]> inputs = (ArrayList<double[]>) in.clone();
		@SuppressWarnings("unchecked")
		ArrayList<double[]> expected = (ArrayList<double[]>) exp.clone();
		ArrayList<double[]> results = new ArrayList<double[]>();

		long startTime = System.currentTimeMillis();
		double totalErrors = 0;

		if (this.config.backPropagationAlgorithm == Config.RPROP_BACKPROPAGATION) {
			return performRPropagation(target, maxEpochs, callBackInterval, listener, inputs, expected, results,
					startTime, totalErrors);
		} else {
			return performStandardBackPropagation(target, maxEpochs, callBackInterval, listener, inputs, expected,
					results, startTime, totalErrors);
		}
	}

	@SuppressWarnings("unchecked")
	public Pair<Integer, Double> optimizeRecurrent(ArrayList<ArrayList<double[]>> in,
			ArrayList<ArrayList<double[]>> exp, double target, int maxEpochs, int callBackInterval,
			OptimizationListener listener) {
		ArrayList<ArrayList<double[]>> expected = (ArrayList<ArrayList<double[]>>) exp.clone();
		ArrayList<double[]> results = new ArrayList<double[]>();

		long startTime = System.currentTimeMillis();
		double totalErrors = 0;

		return performStandardRecurrentBackPropagation(target, maxEpochs, callBackInterval, listener, (ArrayList<ArrayList<double[]>>)in.clone(), expected,
				results, startTime, totalErrors);

	}

	private InputSet saveInputs() {
		InputSet inputSet = new InputSet();
		inputSet.addState(this.allNeurons);
		return inputSet;
	}

	private void restoreInputs(InputSet inputSet) {
		inputSet.restoreState(this.allNeurons);
	}

	private Pair<Integer, Double> performStandardRecurrentBackPropagation(double target, int maxEpochs,
			int callBackInterval, OptimizationListener listener, ArrayList<ArrayList<double[]>> inputs,
			ArrayList<ArrayList<double[]>> expected, ArrayList<double[]> results, long startTime, double totalErrors) {

		Random rnd = new Random();
		rnd.setSeed(1234567);
		int i;
		ArrayList<double[]> expectedResults = new ArrayList<double[]>();
		ArrayList<InputSet> inputSetArray = new ArrayList<InputSet>();

		for (ArrayList<double[]> ex : expected) {
			for (double[] ex2 : ex) {
				expectedResults.add(ex2);
			}
		}

		for (i = 0; i < maxEpochs; i++) {
			// System.out.println("adjusting weights.");

			shuffleArray(rnd, inputs, expected);
			results.clear();

			int index = 0;
			// System.out.println("Epoch " + i);
			for (ArrayList<double[]> inputSet : inputs) {
				inputSetArray.clear();
				ArrayList<double[]> outputSet = expected.get(index);
				int i2 = 0;

				resetRecurrentStates();

				double[] output = feed(inputSet.get(i2));
				inputSetArray.add(saveInputs());
				updatePreviousOutputs();
				results.add(output);

				for (i2 = 1; i2 < inputSet.size(); i2++) {
					output = feed(inputSet.get(i2));
					inputSetArray.add(saveInputs());
					updatePreviousOutputs();
					results.add(output);
				}

				i2 = inputSet.size() - 1;
				inputSetArray.get(i2).restoreState(this.allNeurons);
				this.adjustWeights(outputSet.get(i2), true);

				for (i2 = inputSet.size() - 2; i2 >= 0; i2--) {
					inputSetArray.get(i2).restoreState(this.allNeurons);
					this.adjustWeights(outputSet.get(i2), true);
				}

				// apply all deltas
				// OutputUtils.print(dumpDeltas());
				updateDeltas();
				// System.out.println("update=>\n" + saveStateToJson());
				index++;
			}

			totalErrors = getTotalError(expectedResults, results);

			if (totalErrors < target && (i > 10)) {
				return new Pair<Integer, Double>(i + 1, totalErrors);
			}

			if (i % callBackInterval == 0) {
				long endTime = System.currentTimeMillis();
				long elapsed = endTime - startTime;
				startTime = System.currentTimeMillis();

				if (listener != null) {
					listener.checkpoint(i, totalErrors, elapsed);
				}

			}
		}
		return new Pair<Integer, Double>(i + 1, totalErrors);
	}

	private Pair<Integer, Double> performStandardBackPropagation(double target, int maxEpochs, int callBackInterval,
			OptimizationListener listener, ArrayList<double[]> inputs, ArrayList<double[]> expected,
			ArrayList<double[]> results, long startTime, double totalErrors) {

		Random rnd = new Random();
		rnd.setSeed(1234567);
		int i;
		for (i = 0; i < maxEpochs; i++) {
			// System.out.println("adjusting weights.");
			results.clear();
			shuffleArray(rnd, inputs, expected);

			int index = 0;

			for (double[] input : inputs) {
				double[] output = feed(input);
				results.add(output);
				adjustWeights(expected.get(index++));
			}

			totalErrors = getTotalError(expected, results);

			if (totalErrors < target && (i > 10)) {
				return new Pair<Integer, Double>(i + 1, totalErrors);
			}

			if (i % callBackInterval == 0) {
				long endTime = System.currentTimeMillis();
				long elapsed = endTime - startTime;
				startTime = System.currentTimeMillis();

				if (listener != null) {
					listener.checkpoint(i, totalErrors, elapsed);
				}

			}
		}
		return new Pair<Integer, Double>(i + 1, totalErrors);
	}

	private Pair<Integer, Double> performRPropagation(double target, int maxEpochs, int callBackInterval,
			OptimizationListener listener, ArrayList<double[]> inputs, ArrayList<double[]> expected,
			ArrayList<double[]> results, long startTime, double totalErrors) {
		Random rnd = new Random();
		rnd.setSeed(1234567);
		int i;
		resetAllDeltas();
		for (i = 0; i < maxEpochs; i++) {
			// System.out.println("adjusting weights.");
			results.clear();
			resetDeltas();

			shuffleArray(rnd, inputs, expected);

			int index = 0;

			for (double[] input : inputs) {
				double[] output = feed(input);
				results.add(output);
				computeGradients(expected.get(index++));
			}

			updateWeightsRprop();

			totalErrors = getTotalError(expected, results);

			if (totalErrors < target) {
				return new Pair<Integer, Double>(i + 1, totalErrors);
			}

			if (i % callBackInterval == 0) {
				long endTime = System.currentTimeMillis();
				long elapsed = endTime - startTime;
				startTime = System.currentTimeMillis();

				if (listener != null) {
					listener.checkpoint(i, totalErrors, elapsed);
				}

			}
		}
		return new Pair<Integer, Double>(i + 1, totalErrors);
	}

	private double getTotalError(ArrayList<double[]> expected, ArrayList<double[]> results) {
		double totalErrors = 0;
		int index = 0;
		for (double[] output : results) {
			totalErrors += computeAcccuracy(output, expected.get(index++));
		}

		return totalErrors / results.size();
	}

	private void resetDeltas() {
		for (ArrayList<Neuron> layer : this.layers) {
			for (Neuron n : layer) {
				n.resetDeltas();
			}
		}
	}

	private void resetAllDeltas() {
		for (ArrayList<Neuron> layer : this.layers) {
			for (Neuron n : layer) {
				n.resetAllDeltas();
			}
		}
	}

	private void updateDeltas() {
		for (Neuron n : this.allNeurons) {
			n.applyDelta();
		}
	}

	private void updateWeightsRprop() {
		for (int index = 0; index < this.layerCount - 1; index++) {
			for (Neuron n : this.layers.get(index)) {
				n.updateGradient();
			}
		}
	}

	public String round(double val) {
		DecimalFormat df = new DecimalFormat("#.##########");
		df.setRoundingMode(RoundingMode.CEILING);
		return df.format(val);
	}

	public void saveToFile(String filename) {
		try {
			FileWriter writer = new FileWriter(filename);
			writer.write(saveStateToJson(false));
			writer.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
