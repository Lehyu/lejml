/*
 * Copyright (C) 2018 Baidu, Inc. All Rights Reserved.
 */

// created by lianghongyu at 2018/11/7

package com.lehyu.lejml.loss.impl;

import org.nd4j.linalg.api.ndarray.INDArray;

import com.lehyu.lejml.loss.ILoss;

/**
 * loss: -{t*ln(y)+(1-t)*ln(1-y)}
 * derive: (y-t)*x
 */
public class SoftmaxLoss implements ILoss {
    @Override
    public double computeLoss(INDArray X, INDArray y, INDArray W) {
        return 0;
    }

    @Override
    public INDArray derive(INDArray X, INDArray y, INDArray W) {
        return null;
    }
}
