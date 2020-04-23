/*
 * Copyright (C) 2018 Baidu, Inc. All Rights Reserved.
 */

// created by Lehyu at 2018/10/25

package com.lehyu.lejml.optimizers;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface IOptimizer {

    INDArray optim(INDArray X, INDArray y, INDArray W);

    void setLoss(String name);
}
