/*
 * Copyright (C) 2018 Baidu, Inc. All Rights Reserved.
 */

// created by Lehyu at 2018/11/22

package com.lehyu.lejml.preprocessing;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface IScaler {

    void fit(INDArray X);

    INDArray transform(INDArray X);

    INDArray fitTransform(INDArray X);
}
