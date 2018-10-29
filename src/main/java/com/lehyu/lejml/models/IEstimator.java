/*
 * Copyright (C) 2018 Baidu, Inc. All Rights Reserved.
 */

// created by lianghongyu at 2018/10/25

package com.lehyu.lejml.models;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface IEstimator {
    void fit(INDArray X, INDArray y);

    INDArray predict(INDArray X);
}
