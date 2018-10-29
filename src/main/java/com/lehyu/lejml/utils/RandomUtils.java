/*
 * Copyright (C) 2018 Baidu, Inc. All Rights Reserved.
 */

// created by lianghongyu at 2018/10/25

package com.lehyu.lejml.utils;

import java.util.Random;

import org.nd4j.linalg.api.ndarray.INDArray;

public class RandomUtils {

    public static void shuffle(INDArray X, INDArray y) {
        int[] indices = shuffle(X.rows());
        X = X.getRows(indices);
        y = y.getRows(indices);
    }

    public static int[] shuffle(int nSamples) {
        int[] data = new int[nSamples];
        for (int index = 0; index < nSamples; index++) {
            data[index] = index;
        }
        Random rand = new Random();
        for (int index=nSamples-1; index>1; index--) {
            int tmp = data[index-1];
            int jndex = rand.nextInt(index);
            data[index-1] = data[jndex];
            data[jndex] = tmp;
        }
        return data;
    }
}
