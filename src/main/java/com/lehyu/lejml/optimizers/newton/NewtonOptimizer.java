/*
 * Copyright (C) 2018 Baidu, Inc. All Rights Reserved.
 */

// created by lianghongyu at 2018/10/26

package com.lehyu.lejml.optimizers.newton;

import org.nd4j.linalg.api.ndarray.INDArray;

import com.lehyu.lejml.optimizers.IOptimizer;

public class NewtonOptimizer implements IOptimizer {

    @Override
    public INDArray optim(INDArray X, INDArray y, INDArray W) {
        return null;
    }

    @Override
    public void setLoss(String name) {

    }
}
