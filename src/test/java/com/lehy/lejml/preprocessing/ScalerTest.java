/*
 * Copyright (C) 2018 Baidu, Inc. All Rights Reserved.
 */

// created by Lehyu at 2018/11/22

package com.lehy.lejml.preprocessing;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.lehyu.lejml.preprocessing.impl.MinMaxScaler;
import com.lehyu.lejml.preprocessing.impl.StandardScaler;

public class ScalerTest {

    public static void main(String[] args) {
        INDArray X = Nd4j.rand(10, 3);
        System.out.println(X);
        System.out.println(new MinMaxScaler().fitTransform(X));
        System.out.println(new StandardScaler().fitTransform(X));
    }
}
