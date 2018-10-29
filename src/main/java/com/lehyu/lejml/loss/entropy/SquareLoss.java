/*
 * Copyright (C) 2018 Baidu, Inc. All Rights Reserved.
 */

// created by lianghongyu at 2018/10/29

package com.lehyu.lejml.loss.entropy;

import org.nd4j.linalg.api.ndarray.INDArray;

import com.lehyu.lejml.loss.ILoss;

public class SquareLoss implements ILoss {
    @Override
    public double computeLoss(INDArray X, INDArray y, INDArray W) {
        return 0;
    }

    @Override
    public INDArray derive(INDArray X, INDArray y, INDArray W) {
        return null;
    }
}
