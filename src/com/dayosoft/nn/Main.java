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
		testSoftMax();
	}

	private static double[] colorValue(String colorValue) {
		return new double[] {
		     Integer.parseInt(colorValue.substring(0, 2), 16) / 256.0,
		     Integer.parseInt(colorValue.substring(2, 4), 16) / 256.0,
		     Integer.parseInt(colorValue.substring(4, 6), 16) / 256.0
		};
	}
	
	private static String toColor(double[] output) {
		double maxValue = 0;
		int index = 0;
		for(int i =0; i < output.length; i++) {
			if (output[i] > maxValue) {
				maxValue = output[i];
				index = i;
			}
		}
		switch(index) {
		case 0: return "RED";
		case 1: return "GREEN";
		case 2: return "BLUE";
		}
		return ""; 
	}
	
	private static void testSoftMax() {
		NeuralNet.Config config = new NeuralNet.Config(3, 3, 4);
		config.bias = 1f;
		config.outputBias = 1f;
		config.learningRate = 0.01f;
		config.neuronsPerLayer = 3;
		config.momentumFactor = 0f;
		config.activationFunctionType = Neuron.HTAN;
		config.outputActivationFunctionType = Neuron.SOFTMAX;
		config.errorFormula = Config.CROSS_ENTROPY;
		config.backPropagationAlgorithm = Config.STANDARD_BACKPROPAGATION;
		
		ArrayList<double[]> inputs = new ArrayList<double[]>();
		ArrayList<double[]> expected = new ArrayList<double[]>();
//		  [color_value('E32636'), RED],
//		  [color_value('8B0000'), RED],
//		  [color_value('800000'), RED],
//		  [color_value('65000B'), RED],
//		  [color_value('674846'), RED],
//
//		  #green
//		  [color_value('8F9779'), GREEN],
//		  [color_value('568203'), GREEN],
//		  [color_value('013220'), GREEN],
//		  [color_value('00FF00'), GREEN],
//		  [color_value('006400'), GREEN],
//		  [color_value('00A877'), GREEN],
//
//		  #blue
//		  [color_value('89CFF0'), BLUE],
//		  [color_value('ADD8E6'), BLUE],
//		  [color_value('0000FF'), BLUE],
//		  [color_value('0070BB'), BLUE],
//		  [color_value('545AA7'), BLUE],
//		  [color_value('4C516D'), BLUE],
		inputs.add(colorValue("e32636"));
		inputs.add(colorValue("8b0000"));
		inputs.add(colorValue("674846"));
		
		inputs.add(colorValue("8f9779"));
		inputs.add(colorValue("00ff00"));
		inputs.add(colorValue("006400"));
		inputs.add(colorValue("013220"));
		
		inputs.add(colorValue("89cff0"));
		inputs.add(colorValue("0070bb"));
		inputs.add(colorValue("4c516d"));
		
		final double[] RED = new double[] { 1.0, 0.0, 0.0 };
		final double[] GREEN = new double[] { 0.0, 1.0, 0.0 };
		final double[] BLUE = new double[] { 0.0, 0.0, 1.0 };
		
		expected.add(RED);
		expected.add(RED);
		expected.add(RED);
		
		expected.add(GREEN);
		expected.add(GREEN);
		expected.add(GREEN);
		expected.add(GREEN);

		expected.add(BLUE);
		expected.add(BLUE);
		expected.add(BLUE);
		
		
		ArrayList<double[]> testInputs = new ArrayList<double[]>();
		testInputs.add(colorValue("333399")); //blue
		testInputs.add(colorValue("c80815")); //red
		testInputs.add(colorValue("4c516d")); //blue
		testInputs.add(colorValue("009E60")); //green
		testInputs.add(colorValue("00FF00")); //green
		
		NeuralNet nn = new NeuralNet(config);
		nn.randomizeWeights(-1, 1);

		for(double[] inp : testInputs) {
			double[] out = nn.feed(inp);
			OutputUtils.print(nn.feed(inp));
			System.out.println(toColor(out));
		}
		
		System.out.println("=======training start===========");
		nn.optimize(inputs, expected, 0.1, 1000000, 1000, new OptimizationListener() {

			@Override
			public void checkpoint(int i, double totalErrors, long elapsedPerEpoch) {
				// TODO Auto-generated method stub
//				System.out.println(i + " " + totalErrors + " " + elapsedPerEpoch);
			} 
			
		});
		
		for(double[] inp : testInputs) {
			double[] out = nn.feed(inp);
			OutputUtils.print(nn.feed(inp));
			System.out.println(toColor(out));
		}
	}
	
	private static void testSinRecurrent() {
		System.out.println("starting binary test ....");
		System.out.println("=========================");
		NeuralNet.Config config = new NeuralNet.Config(1, 1, 4);
		config.bias = 1f;
		config.outputBias = 1f;
		config.learningRate = 0.01f;
		config.neuronsPerLayer = 3;
		config.momentumFactor = 0f;
		config.activationFunctionType = Neuron.HTAN;
		config.outputActivationFunctionType = Neuron.HTAN;
		config.isRecurrent = true;
		config.errorFormula = Config.MEAN_SQUARED;
		config.backPropagationAlgorithm = Config.STANDARD_BACKPROPAGATION;
		
		NeuralNet nn = new NeuralNet(config);
		nn.randomizeWeights(-1, 1);
		
		ArrayList<ArrayList<double[]>> lines = new ArrayList<ArrayList<double[]>>();
		ArrayList<ArrayList<double[]>> outputLines = new ArrayList<ArrayList<double[]>>();

		createExample(0, 10, lines, outputLines);
		
		ArrayList<double[]> test2 = new ArrayList<double[]>();
		ArrayList<double[]> testResult2 = new ArrayList<double[]>();
		for(int t = 0; t < 20; t++) {
			test2.add(new double[] {Math.sin(t)});
		}
	
		for(int t = 0; t < 20; t++) {
			testResult2.add(new double[] {Math.sin(t + 1)});
		}
		
		ArrayList<double[]> item3 = new ArrayList<double[]>();
		ArrayList<double[]> itemResult3 = new ArrayList<double[]>();
		for(int t = 90; t < 110; t++) {
			item3.add(new double[] {Math.sin(t)});
		}
		
		for(int t = 90; t < 110; t++) {
			itemResult3.add(new double[] {Math.sin(t + 1)});
		}
		
		OutputUtils.print(nn.feed(test2));
		OutputUtils.print(testResult2);
			
		OutputUtils.print(nn.feed(item3));
		OutputUtils.print(itemResult3);
		
		System.out.println("=======training start===========");
		nn.optimizeRecurrent(lines, outputLines, 0.0001, 0, 1000000, 1000, new OptimizationListener() {

			@Override
			public void checkpoint(int i, double totalErrors, long elapsedPerEpoch) {
				// TODO Auto-generated method stub
				System.out.println(i + " " + totalErrors + " " + elapsedPerEpoch);
			} 
			
		});
		System.out.println("=======training end===========");
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
		
		inputs.add(new double[] { 0.7f});
		expected.add(new double[] { Math.sin(0.7f) } );
		
		inputs.add(new double[] { 0f});
		expected.add(new double[] { Math.sin(0f) } );
		
		Pair<Integer, Double> stats = nn.optimize(inputs, expected, 0.0001f, 1000000, 10, new OptimizationListener() {

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
		showResult(nn, 0.78f);
	}
	
	
	
	private static void showResult(NeuralNet nn, double val) {
		double[] input = new double[] { val };
		double[] result = nn.feed(input);
		System.out.println(result[0] + " = " + Math.sin(val));
	}
}
