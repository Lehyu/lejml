/*
 * Copyright (C) 2018 Baidu, Inc. All Rights Reserved.
 */

// created by lianghongyu at 2018/11/22

package com.lehy.lejml.models.regression;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.api.rng.distribution.impl.NormalDistribution;
import org.nd4j.linalg.factory.Nd4j;

import com.lehyu.lejml.loss.LossUtils;
import com.lehyu.lejml.metrics.impl.MeanSquareError;
import com.lehyu.lejml.models.regression.linear.impl.LinearRegression;
import com.lehyu.lejml.optimizers.sgd.SGDOptimizer;

public class LinearRegressionTest {

    public static void main(String[] args) {
        int rows = 100, cols = 10;
        INDArray X = Nd4j.rand(rows, cols);
        INDArray W = Nd4j.rand(cols, 1);
        Distribution dist = new NormalDistribution();
        int[] shape = {rows, 1};
        INDArray y = X.mmul(W).add(Nd4j.rand(shape, dist));
        System.out.println(W);
        SGDOptimizer optimizer = new SGDOptimizer.Builder().batch(20).eta(1e-3).maxIter(10000).loss(LossUtils.LossEnum
                .SQUARE_LOSS.getName()).lambda(0).earlyStopping(10).build();
        LinearRegression lr = new LinearRegression.Builder().fitIntercept(false).normalize(false).optimizer(optimizer)
                .build();
        lr.fit(X, y);
        INDArray preds = lr.predict(X);
        System.out.println(y);
        System.out.println(preds);
        System.out.println(MeanSquareError.compute(y, preds));
    }
}
