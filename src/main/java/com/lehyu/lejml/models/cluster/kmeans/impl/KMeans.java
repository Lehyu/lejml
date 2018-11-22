/*
 * Copyright (C) 2018 Baidu, Inc. All Rights Reserved.
 */

// created by lianghongyu at 2018/11/22

package com.lehyu.lejml.models.cluster.kmeans.impl;

import org.nd4j.linalg.api.ndarray.INDArray;

import com.lehyu.lejml.models.cluster.kmeans.IKMeans;

// https://stats.stackexchange.com/questions/133656/how-to-understand-the-drawbacks-of-k-means
public class KMeans implements IKMeans {
    @Override
    public void fit(INDArray X, INDArray y) {

    }

    @Override
    public INDArray predict(INDArray X) {
        return null;
    }
}
