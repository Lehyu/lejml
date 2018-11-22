/*
 * Copyright (C) 2018 Baidu, Inc. All Rights Reserved.
 */

// created by lianghongyu at 2018/11/8

package com.lehyu.lejml.metrics.impl;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Accuracy {
    public static double compute(INDArray yTrue, INDArray yPred) {
        assert yTrue.rows() == yPred.rows() : "yTrue and yPred don't have the same dimension";
        double count = 0;
        for (int index = 0; index < yTrue.rows(); index++) {
            if (yTrue.getDouble(index) == yPred.getDouble(index)) {
                count++;
            }
        }
        return count/yTrue.rows();
    }
}
