package com.dayosoft.nn;


import java.util.ArrayList;

import com.dayosoft.nn.NeuralNet;
import com.dayosoft.nn.NeuralNet.Config;
import com.dayosoft.nn.NeuralNet.Pair;
import com.dayosoft.nn.Neuron;
import com.dayosoft.nn.utils.OutputUtils;

public class Main {
	public static void main(String args[]) {
		testBinary();
	}

	private static void testBinary() {
		NeuralNet.Config config = new NeuralNet.Config(1, 1, 2);
		config.bias = 1f;
		config.outputBias = 1f;
		config.learningRate = 1f;
		config.neuronsPerLayer = 1;
		config.momentumFactor = 0f;
		config.activationFunctionType = Neuron.HTAN;
		config.outputActivationFunctionType = Neuron.SIGMOID;
		config.isRecurrent = true;
		config.errorFormula = Config.MEAN_SQUARED;
		config.backPropagationAlgorithm = Config.STANDARD_BACKPROPAGATION;
		
		NeuralNet nn = new NeuralNet(config);
		nn.randomizeWeights(-1, 1);
		
		ArrayList<ArrayList<double[]>> inputs = new ArrayList<ArrayList<double[]>>();
		ArrayList<double[]> expected = new ArrayList<double[]>();
		
		ArrayList<ArrayList<double[]>> lines = new ArrayList<ArrayList<double[]>>();
		
		ArrayList<double[]> item = new ArrayList<double[]>();
		item.add(new double[] {0.1f});
		item.add(new double[] {0.9f});
		item.add(new double[] {0.1f});
		lines.add(item);
		expected.add(new double[] {0.2f} );
		
		
		ArrayList<double[]> item2 = new ArrayList<double[]>();
		item2.add(new double[] {0.1f});
		item2.add(new double[] {0.9f});
		item2.add(new double[] {0.9f});
		lines.add(item2);
		expected.add(new double[] {0.3f});
		
		OutputUtils.print(nn.feed(item));
		OutputUtils.print(nn.feed(item2));
		
		nn.optimizeRecurrent(lines, expected, 0.001, 10000, 100, new OptimizationListener() {

			@Override
			public void checkpoint(int i, double totalErrors, long elapsedPerEpoch) {
				// TODO Auto-generated method stub
				System.out.println(i + " " + totalErrors + " " + elapsedPerEpoch);
			} 
			
		});
		
		OutputUtils.print(nn.feed(item));
		OutputUtils.print(nn.feed(item2));
	}

	private static void testSin() {
		NeuralNet.Config config = new NeuralNet.Config(1, 1, 2);
		config.bias = 1f;
		config.outputBias = 1f;
		config.learningRate = 1f;
		config.neuronsPerLayer = 1;
		config.momentumFactor = 0f;
		config.activationFunctionType = Neuron.SIGMOID;
		config.outputActivationFunctionType = Neuron.SIGMOID;
		config.errorFormula = Config.MEAN_SQUARED;
		config.backPropagationAlgorithm = Config.STANDARD_BACKPROPAGATION;

		NeuralNet nn = new NeuralNet(config);
		nn.randomizeWeights(-1, 1);
		
		ArrayList<double[]> inputs = new ArrayList<double[]>();
		ArrayList<double[]> expected = new ArrayList<double[]>();

		inputs.add(new double[] { 1f});
		expected.add(new double[] { Math.sin(1f) } );
		
		inputs.add(new double[] { 0.9f});
		expected.add(new double[] { Math.sin(0.9f) } );
		
		inputs.add(new double[] { 0.5f});
		expected.add(new double[] { Math.sin(0.5f) } );
		
		
		inputs.add(new double[] { 0.3f});
		expected.add(new double[] { Math.sin(0.3f) } );
		
		inputs.add(new double[] { 0.1f});
		expected.add(new double[] { Math.sin(0.1f) } );
		
		inputs.add(new double[] { 0.2f});
		expected.add(new double[] { Math.sin(0.2f) } );
		
		inputs.add(new double[] { 0.8f});
		expected.add(new double[] { Math.sin(0.8f) } );
		
		inputs.add(new double[] { 0f});
		expected.add(new double[] { Math.sin(0f) } );
		
		Pair<Integer, Double> stats = nn.optimize(inputs, expected, 0.001f, 100000, 1, new OptimizationListener() {

			@Override
			public void checkpoint(int i, double totalErrors, long elapsedPerEpoch) {
//				System.out.println(totalErrors+ " " + elapsedPerEpoch);
			}
			
		});
		System.out.println("optimization run complete " + stats.first() + " " + stats.second());
		System.out.println("test run ....");
		showResult(nn, 0.1f);
		showResult(nn, 0.5f);
		showResult(nn, 0.6f);
		showResult(nn, 0.8f);
	}
	
	
	
	private static void showResult(NeuralNet nn, double val) {
		double[] input = new double[] { val };
		double[] result = nn.feed(input);
		System.out.println(result[0] + " = " + Math.sin(val));
	}
}
