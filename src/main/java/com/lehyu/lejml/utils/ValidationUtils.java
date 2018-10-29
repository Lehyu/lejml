/*
 * Copyright (C) 2018 Baidu, Inc. All Rights Reserved.
 */

// created by lianghongyu at 2018/10/25

package com.lehyu.lejml.utils;

import org.nd4j.linalg.api.ndarray.INDArray;

public class ValidationUtils {

    public static void checkXy(INDArray X, INDArray y) {
        assert null != X : "X should not be null";
        assert null != y : "y should not be null";
        assert X.rows() == y.rows() : "X and y should have the same size of samples";
    }
}
