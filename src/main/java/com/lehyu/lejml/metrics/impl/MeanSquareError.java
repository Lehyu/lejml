/*
 * Copyright (C) 2018 Baidu, Inc. All Rights Reserved.
 */

// created by lianghongyu at 2018/11/22

package com.lehyu.lejml.metrics.impl;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class MeanSquareError {

    public static double compute(INDArray yTrue, INDArray yPred) {
        return Nd4j.sum(Transforms.pow(yPred.sub(yTrue), 2)).getDouble(0)/2.0;
    }
}
