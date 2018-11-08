/*
 * Copyright (C) 2018 Baidu, Inc. All Rights Reserved.
 */

// created by lianghongyu at 2018/10/25

package com.lehyu.lejml.loss;

import com.lehyu.lejml.loss.impl.LogLoss;
import com.lehyu.lejml.loss.impl.SoftmaxLoss;
import com.lehyu.lejml.loss.impl.SquareLoss;

public class LossUtils {
    public enum LossEnum {
        LOG_LOSS(0, "logloss"),
        SQUARE_LOSS(1, "square"),
        SOFTMAX(2, "softmax");


        private int code;
        private String name;

        LossEnum(int code, String name) {
            this.code = code;
            this.name = name;
        }

        public int getCode() {
            return code;
        }

        public void setCode(int code) {
            this.code = code;
        }

        public String getName() {
            return name;
        }

        public void setName(String name) {
            this.name = name;
        }
    }

    public static ILoss getLossByName(String name) {
        if (name.equalsIgnoreCase(LossEnum.LOG_LOSS.getName())) {
            return new LogLoss();
        } else if (name.equalsIgnoreCase(LossEnum.SQUARE_LOSS.getName())) {
            return new SquareLoss();
        } else if (name.equalsIgnoreCase(LossEnum.SOFTMAX.getName())) {
            return new SoftmaxLoss();
        }
        return null;
    }
}
