package com.dayosoft.nn;


import java.util.ArrayList;

import com.dayosoft.nn.NeuralNet;
import com.dayosoft.nn.NeuralNet.Config;
import com.dayosoft.nn.NeuralNet.Pair;
import com.dayosoft.nn.Neuron;
import com.dayosoft.nn.utils.OutputUtils;

public class Main {
	public static void main(String args[]) {
		testSin();
		testSinRecurrent();
	}

	private static void testSinRecurrent() {
		System.out.println("starting binary test ....");
		System.out.println("=========================");
		NeuralNet.Config config = new NeuralNet.Config(1, 1, 7);
		config.bias = 1f;
		config.outputBias = 1f;
		config.learningRate = 0.01f;
		config.momentumFactor = 0f;
		config.activationFunctionType = Neuron.HTAN;
		config.outputActivationFunctionType = Neuron.SIGMOID;
		config.isRecurrent = true;
		config.errorFormula = Config.MEAN_SQUARED;
		config.backPropagationAlgorithm = Config.STANDARD_BACKPROPAGATION;
		
		NeuralNet nn = new NeuralNet(config);
		nn.randomizeWeights(-1, 1);
		
		ArrayList<ArrayList<double[]>> lines = new ArrayList<ArrayList<double[]>>();
		ArrayList<ArrayList<double[]>> outputLines = new ArrayList<ArrayList<double[]>>();

		
		createExample(0, 3, lines, outputLines);
		createExample(4, 9, lines, outputLines);
		createExample(3, 11, lines, outputLines);
		createExample(100, 106, lines, outputLines);
		createExample(20, 27, lines, outputLines);
		createExample(17, 20, lines, outputLines);
		
		ArrayList<double[]> test2 = new ArrayList<double[]>();
		ArrayList<double[]> testResult2 = new ArrayList<double[]>();
		for(int t = 10; t < 14; t++) {
			test2.add(new double[] {Math.sin(t)});
		}
	
		for(int t = 10; t < 14; t++) {
			testResult2.add(new double[] {Math.sin(t + 1)});
		}
		
		ArrayList<double[]> item3 = new ArrayList<double[]>();
		ArrayList<double[]> itemResult3 = new ArrayList<double[]>();
		for(int t = 0; t < 13; t++) {
			item3.add(new double[] {Math.sin(t)});
		}
		
		for(int t = 0; t < 13; t++) {
			itemResult3.add(new double[] {Math.sin(t + 1)});
		}
		
		OutputUtils.print(nn.feed(test2));
		OutputUtils.print(testResult2);
			
		OutputUtils.print(nn.feed(item3));
		OutputUtils.print(itemResult3);
		
		System.out.println(nn.saveStateToJson());
		System.out.println("=======training start===========");
		nn.optimizeRecurrent(lines, outputLines, 0.01, 1000000, 100000, new OptimizationListener() {

			@Override
			public void checkpoint(int i, double totalErrors, long elapsedPerEpoch) {
				// TODO Auto-generated method stub
				System.out.println(i + " " + totalErrors + " " + elapsedPerEpoch);
			} 
			
		});
		
		OutputUtils.print(nn.feed(test2));
		OutputUtils.print(testResult2);
			
		OutputUtils.print(nn.feed(item3));
		OutputUtils.print(itemResult3);
	
		
		System.out.println();
		System.out.println(nn.saveStateToJson());
	}

	private static ArrayList<double[]> createExample(int start, int end, ArrayList<ArrayList<double[]>> lines,
			ArrayList<ArrayList<double[]>> outputLines) {
		ArrayList<double[]> item = new ArrayList<double[]>();
		ArrayList<double[]> itemResult = new ArrayList<double[]>();
		for(int t = start; t < end; t++) {
			item.add(new double[] {Math.sin(t)});
		}
		lines.add(item);
		
		for(int t = start; t < end; t++) {
			itemResult.add(new double[] {Math.sin(t + 1)});
		}
		outputLines.add(itemResult);
		return item;
	}

	private static void testSin() {
		System.out.println("starting sin test ....");
		System.out.println("=========================");
		NeuralNet.Config config = new NeuralNet.Config(1, 1, 4);
		config.bias = 1f;
		config.outputBias = 1f;
		config.learningRate = 1f;
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
