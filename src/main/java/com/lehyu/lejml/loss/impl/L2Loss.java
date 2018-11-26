/*
 * Copyright (C) 2018 Baidu, Inc. All Rights Reserved.
 */

// created by lianghongyu at 2018/10/25

package com.lehyu.lejml.loss.impl;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class L2Loss {

    public static double computeLoss(INDArray W) {
        INDArray res = Transforms.pow(W, 2);
        return Nd4j.sum(res).getDouble(0);
    }

    public static INDArray derive(INDArray W) {
        return W;
    }
}
