/*
 * Copyright (C) 2018 Baidu, Inc. All Rights Reserved.
 */

// created by lianghongyu at 2018/10/25

package com.lehy.lejml.models.classifier.logistics;

import java.io.IOException;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.lehyu.lejml.loss.LossUtils;
import com.lehyu.lejml.models.linear_model.classifier.logistics.impl.LogisticsRegression;
import com.lehyu.lejml.optimizers.sgd.SGDOptimizer;

public class LogisticsRegressionTest {

    public static void main(String[] args) throws IOException {
        INDArray data = loadIris();
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

    private static INDArray loadIris() throws IOException {
        String path = "input/iris/iris2c.data";
        return Nd4j.readNumpy(path, ",");
    }
}
