package com.dayosoft.nn;


import java.util.ArrayList;

import com.dayosoft.nn.NeuralNet;
import com.dayosoft.nn.NeuralNet.Config;
import com.dayosoft.nn.Neuron;
import com.dayosoft.nn.utils.OutputUtils;

public class Main {
	public static void main(String args[]) {
		NeuralNet.Config config = new NeuralNet.Config(1, 1, 20);
		config.bias = 1f;
		config.outputBias = 1f;
		config.learningRate = 0.1f;
		config.neuronsPerLayer = 5;
		config.momentumFactor = 0.5f;
		config.activationFunctionType = Neuron.HTAN;
		config.outputActivationFunctionType = Neuron.SIGMOID;
		config.errorFormula = Config.MEAN_SQUARED;

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
		
		nn.optimize(inputs, expected, 0.01f, 1000000, 1000, false, null);
		nn.dumpStates(false);

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
