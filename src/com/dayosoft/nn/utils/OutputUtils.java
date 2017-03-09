package com.dayosoft.nn.utils;

import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

public class OutputUtils {
	private static void printOutput(double[] output) {
		for (double o : output) {
			System.out.print(o >= 0.5f ? "1" : "0");
			System.out.print(" ");
		}
		System.out.println();
	}

	private static void printOutputF(double[] output) {
		for (double o : output) {
			System.out.print(round(o));
			System.out.print(" ");
		}
		System.out.println();
	}

	public static String round(double val) {
		DecimalFormat df = new DecimalFormat("#.##########");
		df.setRoundingMode(RoundingMode.CEILING);
		return df.format(val);
	}

	public static void print(ArrayList<ArrayList<double[]>> lines) {
		System.out.println("========== " + lines.size());
		for (ArrayList<double[]> arrs : lines) {
			for (double[] a : arrs) {
				StringBuilder str = new StringBuilder();
				for (double o : a) {
					str.append(round(o));
					str.append(" ");
				}
				System.out.print(str.toString() + " ");
			}
			System.out.println();
		}
		System.out.println("========== ");
	}

	public static void print(List<double[]> arrs) {
		for (double[] a : arrs) {
			StringBuilder str = new StringBuilder();
			for (double o : a) {
				str.append(round(o));
				str.append(" ");
			}
			System.out.print(str.toString() + " ");
		}
		System.out.println();
	}

	public static String printBoolArr(double arr[]) {
		StringBuilder str = new StringBuilder();
		for (double o : arr) {
			str.append(round(o));
			str.append(" ");
		}
		return str.toString();
	}

	public static void printIntArr(int[] totalTime) {
		StringBuilder str = new StringBuilder();
		for (int o : totalTime) {
			str.append(o);
			str.append(" ");
		}
		System.out.println(str.toString());
	}
}
