package com.dayosoft.nn;


import java.util.ArrayList;

import com.dayosoft.nn.NeuralNet;
import com.dayosoft.nn.NeuralNet.Config;
import com.dayosoft.nn.NeuralNet.Pair;
import com.dayosoft.nn.Neuron;
import com.dayosoft.nn.utils.OutputUtils;

public class Main {
	public static void main(String args[]) {
		NeuralNet.Config config = new NeuralNet.Config(1, 1, 2);
		config.bias = 1f;
		config.outputBias = 1f;
		config.learningRate = 1f;
		config.neuronsPerLayer = 1;
		config.momentumFactor = 0f;
		config.activationFunctionType = Neuron.SIGMOID;
		config.outputActivationFunctionType = Neuron.SIGMOID;
		config.errorFormula = Config.MEAN_SQUARED;
		config.backPropagationAlgorithm = Config.RPROP_BACKPROPAGATION;

		NeuralNet nn = new NeuralNet(config);
//		nn.randomizeWeights(-1, 1);
		
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
		
		Pair<Integer, Double> stats = nn.optimize(inputs, expected, 0.01f, 10, 1, new OptimizationListener() {

			@Override
			public void checkpoint(int i, double totalErrors, long elapsedPerEpoch) {
//				System.out.println(totalErrors+ " " + elapsedPerEpoch);
			}
			
		});
		System.out.println("optimization run complete " + stats.first() + " " + stats.second());
		System.out.println("test run ....");
		double[] input = new double[] { 0.21f };
		double[] result = nn.feed(input);
		OutputUtils.print(result);
		
		System.out.println("expected ...." + Math.sin(0.21f));
		
		System.out.println("test run ....");
		input = new double[] { 0.6f };
		result = nn.feed(input);
		OutputUtils.print(result);
		
		System.out.println("expected ...." + Math.sin(0.6f));
	}
}
