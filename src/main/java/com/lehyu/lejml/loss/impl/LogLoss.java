/*
 * Copyright (C) 2018 Baidu, Inc. All Rights Reserved.
 */

// created by lianghongyu at 2018/10/25

package com.lehyu.lejml.loss.impl;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import com.lehyu.lejml.loss.ILoss;

/**
 * loss: -{t*ln(y)+(1-t)*ln(1-y)}
 * derive: (y-t)*x
 */
public class LogLoss implements ILoss {

    @Override
    public double computeLoss(INDArray X, INDArray y, INDArray W) {
        INDArray yHat = Transforms.exp(X.mmul(W).rsub(0)).add(1.0).rdiv(1.0);
        INDArray left = y.mul(Transforms.log(yHat));
        INDArray right = y.rsub(1.0).mul(Transforms.log(yHat.rsub(1.0)));
        INDArray sum = Nd4j.sum(left.add(right));
        return -sum.getDouble(0);
    }

    @Override
    public INDArray derive(INDArray X, INDArray y, INDArray W) {
        INDArray yHat = Transforms.exp(X.mmul(W).rsub(0)).add(1.0).rdiv(1.0);
        return X.transpose().mmul(yHat.sub(y));
    }
}
