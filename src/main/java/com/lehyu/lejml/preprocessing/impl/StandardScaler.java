/*
 * Copyright (C) 2018 Baidu, Inc. All Rights Reserved.
 */

// created by lianghongyu at 2018/11/22

package com.lehyu.lejml.preprocessing.impl;

import org.nd4j.linalg.api.ndarray.INDArray;

import com.lehyu.lejml.preprocessing.IScaler;

public class StandardScaler implements IScaler {
    private INDArray meanVec;
    private INDArray stdVec;
    private boolean isFitted = false;

    @Override
    public void fit(INDArray X) {
        meanVec = X.mean(0);
        stdVec = X.std(0);
        isFitted = true;
    }

    @Override
    public INDArray transform(INDArray X) {
        assert isFitted : "fit model first!";
        return X.subRowVector(meanVec).divRowVector(stdVec);
    }

    @Override
    public INDArray fitTransform(INDArray X) {
        this.fit(X);
        return this.transform(X);
    }
}
