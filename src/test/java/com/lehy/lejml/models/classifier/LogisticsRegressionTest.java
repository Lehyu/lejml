/*
 * Copyright (C) 2018 Baidu, Inc. All Rights Reserved.
 */

// created by lianghongyu at 2018/10/25

package com.lehy.lejml.models.classifier;

import java.io.IOException;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.lehyu.lejml.loss.LossUtils;
import com.lehyu.lejml.metrics.impl.Accuracy;
import com.lehyu.lejml.models.linear_model.classifier.logistics.impl.LogisticsRegression;
import com.lehyu.lejml.optimizers.sgd.SGDOptimizer;

public class LogisticsRegressionTest {

    public static void main(String[] args) throws IOException {
        testSoftmaxLogisticsRegression();
    }

    private static void testSoftmaxLogisticsRegression() throws IOException {
        INDArray data = loadWholeIris();
        INDArray labels = data.getColumn(data.columns() - 1);
        INDArray y = oneHot(labels);
        INDArray X = data.getColumns(0, 1, 2, 3);
        SGDOptimizer optimizer =
                new SGDOptimizer.Builder().batch(5).eta(1e-3).maxIter(10000).loss(LossUtils.LossEnum.SOFTMAX.getName())
                        .lambda(0).earlyStopping(20).build();
        LogisticsRegression lr =
                new LogisticsRegression.Builder().fitIntercept(true).normalize(true).isMultiClass(true).copied(true)
                        .optimizer(optimizer).build();
        lr.fit(X, y);
        INDArray prob = lr.predict(X);
        System.out.println(prob);
        System.out.println(Accuracy.compute(labels, prob));
    }

    private static INDArray oneHot(INDArray y) {
        INDArray yHat = Nd4j.zeros(y.rows(), 3);
        for (int row = 0; row < y.rows(); row++) {
            int col = (int) y.getDouble(row);
            yHat.put(row, col, 1);
        }
        return yHat;
    }

    private static void testBinaryLogisticsRegression() throws IOException {
        INDArray data = loadBinaryIris();
        INDArray y = data.getColumn(data.columns() - 1);
        INDArray X = data.getColumns(0, 1, 2, 3);
        SGDOptimizer optimizer =
                new SGDOptimizer.Builder().batch(20).eta(0.1).maxIter(1000).loss(LossUtils.LossEnum.LOG_LOSS.getName())
                        .lambda(1e-3).earlyStopping(10).build();
        LogisticsRegression lr =
                new LogisticsRegression.Builder().fitIntercept(true).normalize(true).optimizer(optimizer).build();
        lr.fit(X, y);
        INDArray prob = lr.predict(X);
        System.out.println(prob);
    }

    private static INDArray loadIris(String path) throws IOException {
        return Nd4j.readNumpy(path, ",");
    }

    private static INDArray loadWholeIris() throws IOException {
        return Nd4j.readNumpy("input/iris/iris.data", ",");
    }

    private static INDArray loadBinaryIris() throws IOException {
        return Nd4j.readNumpy("input/iris/iris2c.data", ",");
    }

}
