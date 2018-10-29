/*
 * Copyright (C) 2018 Baidu, Inc. All Rights Reserved.
 */

// created by lianghongyu at 2018/10/25

package com.lehyu.lejml.loss;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface ILoss {

    /**
     * compute current loss
     * @param X: (n_samples*n_features)
     * @param y: (n_samples*n_targets)
     * @param W: (n_features*n_targets)
     * @return double
     * **/
    double computeLoss(INDArray X, INDArray y, INDArray W);

    INDArray derive(INDArray X, INDArray y, INDArray W);
}
