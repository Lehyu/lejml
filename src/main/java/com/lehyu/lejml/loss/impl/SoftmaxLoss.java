/*
 * Copyright (C) 2018 Baidu, Inc. All Rights Reserved.
 */

// created by lianghongyu at 2018/11/7

package com.lehyu.lejml.loss.impl;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import com.lehyu.lejml.loss.ILoss;

/**
 * hidden = exp(xw)
 * $$y=\frac{\exp(XW)}{\sum\exp(XW)}$$
 * loss: $$-\sum_{n=1}^N\sum_{k=1}^Kt_{nk}\lny_{nk}$$
 * derive: (y-t)*x
 */
public class SoftmaxLoss implements ILoss {
    @Override
    public double computeLoss(INDArray X, INDArray y, INDArray W) {
        INDArray yHat = Transforms.exp(X.mmul(W));
        yHat = yHat.divColumnVector(yHat.sum(1));
        return -Nd4j.sum(y.mul(Transforms.log(yHat))).getDouble(0);
    }

    @Override
    public INDArray derive(INDArray X, INDArray y, INDArray W) {
        INDArray yHat = Transforms.exp(X.mmul(W));
        yHat = yHat.divColumnVector(y.sum(1));
        INDArray derivation = yHat.sub(y);
        return X.transpose().mmul(derivation);
    }
}
