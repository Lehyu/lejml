/*
 * Copyright (C) 2018 Baidu, Inc. All Rights Reserved.
 */

// created by lianghongyu at 2018/10/29

package com.lehyu.lejml.loss.impl;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import com.lehyu.lejml.loss.ILoss;


/**
 * loss: 1/2 * \sum_{n_1}^N(y_n - t_n)^2
 * derive: (y-t)*x
 */
public class SquareLoss implements ILoss {
    @Override
    public double computeLoss(INDArray X, INDArray y, INDArray W) {
        INDArray yHat = X.mmul(W);
        INDArray res = yHat.sub(y);
        return res.transpose().mmul(res).getDouble(0)/(2.0*X.rows());
    }

    @Override
    public INDArray derive(INDArray X, INDArray y, INDArray W) {
        INDArray yHat = X.mmul(W);
        return X.transpose().mmul(yHat.sub(y)).div(X.rows());
    }
}
