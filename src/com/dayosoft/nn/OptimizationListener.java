package com.dayosoft.nn;

public interface OptimizationListener {

	void checkpoint(int i, double totalErrors);

}
