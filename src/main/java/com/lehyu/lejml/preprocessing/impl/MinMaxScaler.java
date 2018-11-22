/*
 * Copyright (C) 2018 Baidu, Inc. All Rights Reserved.
 */

// created by lianghongyu at 2018/11/22

package com.lehyu.lejml.preprocessing.impl;

import org.nd4j.linalg.api.ndarray.INDArray;

import com.lehyu.lejml.preprocessing.IScaler;

public class MinMaxScaler implements IScaler {
    private INDArray minAttrVec;
    private boolean isFitted = false;
    private INDArray disAttrVec;

    @Override
    public void fit(INDArray X) {
        minAttrVec = X.min(0);
        disAttrVec = X.max(0).sub(minAttrVec);
        isFitted = true;
    }

    @Override
    public INDArray transform(INDArray X) {
        assert isFitted : "fit model first";
        return X.subRowVector(minAttrVec).divRowVector(disAttrVec);
    }

    @Override
    public INDArray fitTransform(INDArray X) {
        this.fit(X);
        return this.transform(X);
    }
}
