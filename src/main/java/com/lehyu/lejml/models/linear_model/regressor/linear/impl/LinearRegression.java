/*
 * Copyright (C) 2018 Baidu, Inc. All Rights Reserved.
 */

// created by lianghongyu at 2018/10/26

package com.lehyu.lejml.models.linear_model.regressor.linear.impl;

import org.nd4j.linalg.api.ndarray.INDArray;

import com.lehyu.lejml.loss.LossUtils;
import com.lehyu.lejml.models.linear_model.LinearModel;
import com.lehyu.lejml.models.linear_model.regressor.linear.ILinearRegression;
import com.lehyu.lejml.utils.ValidationUtils;

public class LinearRegression extends LinearModel implements ILinearRegression {

    public static class Builder extends LinearModel.Builder<Builder> {
        public LinearRegression build() {
            return new LinearRegression(this);
        }
    }

    public LinearRegression(Builder builder) {
        super(builder);
        this.optimizer.setLoss(LossUtils.LossEnum.SQUARE_LOSS.getName());
    }

    @Override
    public void fit(INDArray X, INDArray y) {
        ValidationUtils.checkXy(X, y);
        assert y.columns() == 1 : "Linear regression should have only one target";
        X = this.normalize(X);
        X = this.appendIntercept(X);
        this.initWeights(X.columns(), y.columns());
        X = this.optimizer.optim(X, y, W);
        this.isTrained = true;
    }

    @Override
    public INDArray predict(INDArray X) {
        assert this.isTrained : "Train first";
        X = this.appendIntercept(X);
        return X.mmul(W);
    }
}
