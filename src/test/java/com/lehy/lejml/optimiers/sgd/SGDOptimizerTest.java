/*
 * Copyright (C) 2018 Baidu, Inc. All Rights Reserved.
 */

// created by Lehyu at 2018/10/25

package com.lehy.lejml.optimiers.sgd;

import java.io.IOException;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class SGDOptimizerTest {

    public static void main(String args[]) throws IOException {
        String path = "input/iris/iris2c.data";
        INDArray data = Nd4j.readNumpy(path, ",");
        data = Nd4j.append(data, 1, 1, -1);
        System.out.println(data);
    }

}
