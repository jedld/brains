package com.dayosoft.nn;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import com.dayosoft.nn.NeuralNet.Pair;
import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonIOException;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.JsonSyntaxException;

public class NeuralNet {
	private static final int MAX_EPOCHS = 10000000;
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

	public static NeuralNet loadStateFromJson(NeuralNet nn, String jsonFile)
			throws JsonIOException, JsonSyntaxException, FileNotFoundException {
		JsonParser parser = new JsonParser();
		JsonElement output = parser.parse(new FileReader(jsonFile));

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

		ArrayList<ArrayList<double[]>> weights = new ArrayList<ArrayList<double[]>>();
		ArrayList<ArrayList<Double>> biasWeights = new ArrayList<ArrayList<Double>>();
		for (int i = 0; i < layers.size(); i++) {
			JsonArray layer = layers.get(i).getAsJsonArray();
			ArrayList<double[]> n = new ArrayList<double[]>();
			ArrayList<Double> layerBiases = new ArrayList<Double>();
			for (int i2 = 0; i2 < layer.size(); i2++) {
				JsonObject neuronObj = layer.get(i2).getAsJsonObject();
				JsonArray weightArr = neuronObj.getAsJsonArray("weights");
				layerBiases.add(neuronObj.get("biasWeight").getAsDouble());
				double w[] = new double[weightArr.size()];
				for (int i3 = 0; i3 < weightArr.size(); i3++) {
					w[i3] = weightArr.get(i3).getAsDouble();
				}
				n.add(w);
			}
			biasWeights.add(layerBiases);
			weights.add(n);
		}

		if (nn == null) {
			nn = new NeuralNet(config);
		}

		nn.loadWeights(weights);
		nn.loadBiases(biasWeights);
		return nn;
	}

	public String saveStateToJson() {
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

		JsonArray layerArray = new JsonArray();
		jsonObject.add("neurons", layerArray);

		for (ArrayList<Neuron> layer : layers) {
			JsonArray neuronLayer = new JsonArray();
			for (Neuron n : layer) {
				JsonObject nObj = new JsonObject();
				JsonArray w = new JsonArray();
				for (double d : n.getWeights()) {
					w.add(d);
				}
				nObj.addProperty("bias", n.bias);
				nObj.addProperty("biasWeight", n.biasWeight);
				nObj.add("weights", w);
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
		public static final int CROSS_ENTROPY = 1;
		public static int DERIVATIVE = 1;
		public static int CE = 2;
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

		public Config(int inputCount, int outputCount, int neuronCount) {
			this.inputCount = inputCount;
			this.neuronCount = neuronCount;
			this.outputCount = outputCount;
			this.bias = 0f;
			this.outputBias = 0f;
			this.learningRate = 0.02f;
			this.neuronsPerLayer = 3;
			this.momentumFactor = 0f;
			this.activationFunctionType = Neuron.SIGMOID;
			this.outputActivationFunctionType = Neuron.SIGMOID;
			this.errorFormula = Config.MEAN_SQUARED;
			this.gradientFormula = Config.DERIVATIVE;
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
				System.out.println(" (b:" + n.bias + ") " + n.getTotal() + " = " + n.fire());
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
				System.out.print("n:" + n.getId() + " -> ");
				doubleArr.add(n.getWeights().clone());
				System.out.print(i + " ");
				for (double f : n.getWeights()) {
					System.out.print(f + " ");
				}
				System.out.println();
			}
			list.add(doubleArr);
		}
		return list;
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
			}
		}
	}

	public void adjustWeights(double expectedOutput[], boolean batchLearning) {
		double deltaSum = 0;
		// System.out.println("========== training ============");
		for (int i = layerCount - 1; i >= 0; i--) {
			// System.out.println("----------- Layer " + i + " --------------");
			ArrayList<Neuron> layer = layers.get(i);
			if (i == layerCount - 1) {
				int i2 = 0;
				for (Neuron n : layer) {
					double errorTerm;
					if (this.gradientFormula == Config.DERIVATIVE) {
						errorTerm = n.derivative() * (expectedOutput[i2++] - n.fire());
					} else {
						errorTerm = expectedOutput[i2++] - n.fire();
					}
					double y = n.adjustForOutput(errorTerm, learningRate, momentumFactor, batchLearning);
					deltaSum += y;
				}
			} else {
				double currentDeltaSum = 0;
				for (Neuron n : layer) {
					double errorTerm = n.derivative() * deltaSum;
					double y = n.adjustForOutput(errorTerm, learningRate, momentumFactor, batchLearning);
					currentDeltaSum += y;
				}
				deltaSum = currentDeltaSum;
			}

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

		if (neurons % config.neuronsPerLayer != 0) {
			this.layerCount += 1;
		}

		System.out.println("creating " + this.layerCount + " layers ");
		System.out.println("Learning Rate " + this.learningRate);

		if (this.activationFunctionType == Neuron.HTAN) {
			System.out.println("Using Hyperbolic Tan()");
		} else {
			System.out.println("Using Symboid()");
		}
		for (int i = 0; i < layerCount; i++) {
			ArrayList<Neuron> aLayer = new ArrayList<>();
			Neuron neuron;
			if (i == 0) {
				for (int i2 = 0; i2 < config.neuronsPerLayer; i2++) {
					neuron = new Neuron(i2, i, config.inputCount, config.bias, this.activationFunctionType);
					aLayer.add(neuron);
					allNeurons.add(neuron);
				}
			} else if (i == layerCount - 1) {
				for (int i2 = 0; i2 < outputCount; i2++) {
					neuron = new Neuron(i2, i, config.neuronsPerLayer, config.outputBias,
							this.outputActivationFunctionType);
					aLayer.add(neuron);
					allNeurons.add(neuron);
				}
			} else {
				for (int i2 = 0; i2 < config.neuronsPerLayer; i2++) {
					neuron = new Neuron(i2, i, config.neuronsPerLayer, config.bias, this.activationFunctionType);
					aLayer.add(neuron);
					allNeurons.add(neuron);
				}
			}

			layers.add(aLayer);
		}

	}

	public double[] feed(double inputs[]) {
		double output[] = new double[outputCount];

		for (Neuron n : allNeurons) {
			n.reset();
		}
		ArrayList<Double> currentFireList = null;
		for (int currentLayer = 0; currentLayer < layerCount; currentLayer++) {
			ArrayList<Neuron> thisLayer = layers.get(currentLayer);
			if (currentLayer == 0) {
				ArrayList<Double> fireList = new ArrayList<Double>();
				int index = 0;
				for (double input : inputs) {
					for (Neuron neuron : thisLayer) {
						neuron.setInput(index, input);
					}
					index++;
				}

				for (Neuron neuron : thisLayer) {
					fireList.add(neuron.fire());
				}
				currentFireList = fireList;
			} else if (currentLayer == layerCount - 1) {
				setInputs(currentFireList, thisLayer);
				int index = 0;
				for (Neuron n : thisLayer) {
					output[index++] = n.fire();
				}
			} else {
				ArrayList<Double> fireList = setInputs(currentFireList, thisLayer);
				for (Neuron neuron : thisLayer) {
					fireList.add(neuron.fire());
				}
				currentFireList = fireList;
			}
		}
		return output;
	}

	private ArrayList<Double> setInputs(ArrayList<Double> currentFireList, ArrayList<Neuron> thisLayer) {
		int index = 0;
		ArrayList<Double> fireList = new ArrayList<Double>();
		for (double fOutputs : currentFireList) {
			for (Neuron neuron : thisLayer) {
				neuron.setInput(index, fOutputs);
			}
			index++;
		}
		return fireList;
	}

	public double computeAcccuracy(double output[], double expectedOutput[]) {
		double difference = 0;
		for (int i = 0; i < output.length; i++) {
			if (this.errorFormula == Config.MEAN_SQUARED) {
				difference += 0.5f * Math.pow(expectedOutput[i] - output[i], 2);
			} else if (this.errorFormula == Config.CROSS_ENTROPY) {
				difference += Math.log(output[i]) * expectedOutput[i];
			}
		}
		// if (this.errorFormula == Config.CROSS_ENTROPY) {
		// difference = -1 * difference;
		// }
		return difference;
	}

	// Implementing Fisher–Yates shuffle
	void shuffleArray(ArrayList<double[]> input, ArrayList<double[]> output) {
		Random rnd = ThreadLocalRandom.current();
		for (int i = input.size() - 1; i > 0; i--) {
			int index = rnd.nextInt(i + 1);
			// Simple swap
			swapElement(input, i, index);
			swapElement(output, i, index);
		}
	}

	private void swapElement(ArrayList<double[]> input, int i, int index) {
		double[] a = input.get(index);
		input.set(index, input.get(i));
		input.set(i, a);
	}

	public void optimize(ArrayList<double[]> in, ArrayList<double[]> exp, double target, boolean batchLearning) {
		System.out.println("training started.");
		if (batchLearning) {
			System.out.println("using batch learning mode.");
		}
		saveToFile("temp.json");
		ArrayList<double[]> inputs = (ArrayList<double[]>) in.clone();
		ArrayList<double[]> expected = (ArrayList<double[]>) exp.clone();
		ArrayList<double[]> results = new ArrayList<double[]>();
		dumpStates(false);
		long startTime = System.currentTimeMillis();
		for (int i = 0; i < MAX_EPOCHS; i++) {
			int index = 0;
			// System.out.println("adjusting weights.");
			results.clear();
			shuffleArray(inputs, expected);
			for (double[] input : inputs) {
				double[] output = feed(input);
				adjustWeights(expected.get(index++), batchLearning);
				if (i % 1000 == 0) {
					results.add(output);
				}
			}

			if (batchLearning) {
//				dumpStates(true);
				updateWeights();
				if (i % 1000 == 0) {
					results.clear();
					for (double[] input : inputs) {
						double[] output = feed(input);
						results.add(output);
					}
				}
			}

			if (i % 1000 == 0) {
				long endTime = System.currentTimeMillis();
				long elapsed = endTime - startTime;
				startTime = System.currentTimeMillis();
				double totalErrors = 0;
				index = 0;
				for (double[] output : results) {
					totalErrors += computeAcccuracy(output, expected.get(index++));
				}
				totalErrors = totalErrors / index;
				System.out.println("e = " + round(totalErrors) + " - " + 1000.0f / (elapsed / 1000.0f) + "/s");
				saveToFile("temp.json");
				if (totalErrors < target) {
					break;
				}
			}

		}

		System.out.println("done training.");
		File file = new File("temp.json");
		if (file.exists()) {
			file.delete();
		}
	}

	private void updateWeights() {
		for (ArrayList<Neuron> layer : this.layers) {
			for (Neuron n : layer) {
				n.applyDelta();
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
			writer.write(saveStateToJson());
			writer.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
