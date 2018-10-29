/*
 * Copyright (C) 2018 Baidu, Inc. All Rights Reserved.
 */

// created by lianghongyu at 2018/10/25

package com.lehyu.lejml.models.linear_model.classifier.logistics.impl;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

import com.lehyu.lejml.loss.LossUtils;
import com.lehyu.lejml.models.linear_model.LinearModel;
import com.lehyu.lejml.models.linear_model.classifier.logistics.ILogisticsRegression;
import com.lehyu.lejml.utils.ValidationUtils;

/**
 * Logistics Regression: y = 1/(1+exp(-wx))
 */
public class LogisticsRegression extends LinearModel implements ILogisticsRegression {

    public LogisticsRegression(Builder builder) {
        super(builder);
        this.optimizer.setLoss(LossUtils.LossEnum.LOG_LOSS.getName());
    }

    @Override
    public void fit(INDArray X, INDArray y) {
        ValidationUtils.checkXy(X, y);
        assert y.columns() == 1 : "Logistics regression should have only one target";
        X = this.normalize(X);
        X = this.appendIntercept(X);
        initWeights(X.columns(), y.columns());
        W = optimizer.optim(X, y, W);
        this.isTrained = true;
    }

    @Override
    public INDArray predict(INDArray X) {
        assert this.isTrained : "Train first";
        X = this.appendIntercept(X);
        return Transforms.exp(X.mmul(W).rsub(0)).add(1.0).rdiv(1.0);
    }
}
