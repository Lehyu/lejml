/*
 * Copyright (C) 2018 Baidu, Inc. All Rights Reserved.
 */

// created by Lehyu at 2018/10/29

package com.lehyu.lejml.metrics;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface IMetrics {

    double compute(INDArray yTrue, INDArray yPred);
}
