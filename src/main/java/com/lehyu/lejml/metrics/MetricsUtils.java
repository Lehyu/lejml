/*
 * Copyright (C) 2018 Baidu, Inc. All Rights Reserved.
 */

// created by lianghongyu at 2018/10/29

package com.lehyu.lejml.metrics;

public class MetricsUtils {
    public enum MetricsEnum {
        MSE(0, "mse"),
        MAE(1, "mae"),
        LOG_LOSS(2, "logloss"),
        SOFTMAX(3, "cross"),
        ACCURACY(4, "accuracy");


        private int code;
        private String name;

        MetricsEnum(int code, String name) {
            this.code = code;
            this.name = name;
        }
    }

    public IMetrics getMetricsByName(String name) {
        return null;
    }
}
