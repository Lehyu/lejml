/*
 * Copyright (C) 2018 Baidu, Inc. All Rights Reserved.
 */

// created by Lehyu at 2018/10/25

package com.lehyu.lejml.models.classifier.logistics.impl;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import com.lehyu.lejml.loss.LossUtils;
import com.lehyu.lejml.models.LinearModel;
import com.lehyu.lejml.models.classifier.logistics.ILogisticsRegression;
import com.lehyu.lejml.utils.ValidationUtils;

/**
 * Logistics Regression: y = 1/(1+exp(-wx))
 */
public class LogisticsRegression extends LinearModel implements ILogisticsRegression {

    private boolean isMultiClass;

    public static class Builder extends LinearModel.Builder<Builder> {
        private boolean isMultiClass = false;

        public Builder isMultiClass(boolean isMultiClass) {
            this.isMultiClass = isMultiClass;
            return this;
        }

        public LogisticsRegression build() {
            return new LogisticsRegression(this);
        }
    }

    public LogisticsRegression(Builder builder) {
        super(builder);
        this.isMultiClass = builder.isMultiClass;
        if (isMultiClass) {
            this.optimizer.setLoss(LossUtils.LossEnum.SOFTMAX.getName());
        } else {
            this.optimizer.setLoss(LossUtils.LossEnum.LOG_LOSS.getName());
        }
    }

    @Override
    public void fit(INDArray X, INDArray y) {
        ValidationUtils.checkXy(X, y);
        assert this.isMultiClass || y.columns() == 1 : "Logistics regression should have only one target";
        INDArray XCopy = this.copy(X);
        XCopy = this.normalize(XCopy);
        XCopy = this.appendIntercept(XCopy);
        initWeights(XCopy.columns(), y.columns());
        W = optimizer.optim(XCopy, y, W);
        this.isTrained = true;
    }

    @Override
    public INDArray predict(INDArray X) {
        INDArray y = predictProb(X);
        return Nd4j.argMax(y, 1);
    }


    @Override
    public INDArray predictProb(INDArray X) {
        assert this.isTrained : "Train first";
        INDArray XCopy = this.copy(X);
        XCopy = this.normalize(XCopy);
        XCopy = this.appendIntercept(XCopy);
        INDArray y;
        if (this.isMultiClass){
            y = Transforms.exp(XCopy.mmul(W));
            y = y.divColumnVector(y.sum(1));
        } else {
            y = Transforms.exp(XCopy.mmul(W).rsub(0)).add(1.0).rdiv(1.0);
        }
        return y;
    }
}
