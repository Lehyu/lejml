/*
 * Copyright (C) 2018 Baidu, Inc. All Rights Reserved.
 */

// created by lianghongyu at 2018/10/29

package com.lehyu.lejml.models.linear_model;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import com.lehyu.lejml.optimizers.IOptimizer;
import com.lehyu.lejml.optimizers.sgd.SGDOptimizer;

public class LinearModel {
    protected int nJobs;
    protected boolean normalize;
    protected boolean fitIntercept;
    protected IOptimizer optimizer;
    protected INDArray W;
    protected boolean isTrained;

    public static class Builder {
        private int nJobs = 1;
        private boolean normalize = true;
        private boolean fitIntercept = true;
        private IOptimizer optimizer = new SGDOptimizer();

        public Builder nJobs(int nJobs) {
            this.nJobs = nJobs;
            return this;
        }

        public Builder normalize(boolean normalize) {
            this.normalize = normalize;
            return this;
        }

        public Builder fitIntercept(boolean fitIntercept) {
            this.fitIntercept = fitIntercept;
            return this;
        }

        public Builder optimizer(IOptimizer optimizer) {
            this.optimizer = optimizer;
            return this;
        }

        public LinearModel build() {
            return new LinearModel(this);
        }
    }

    public LinearModel(Builder builder) {
        this.fitIntercept = builder.fitIntercept;
        this.normalize = builder.normalize;
        this.optimizer = builder.optimizer;
        this.nJobs = builder.nJobs;
    }

    protected INDArray normalize(INDArray X) {
        return this.normalize ? Transforms.normalizeZeroMeanAndUnitVariance(X) : X;
    }

    protected INDArray appendIntercept(INDArray X) {
        return this.fitIntercept ? X = Nd4j.append(X, 1, 1, -1) : X;

    }

    protected void initWeights(int nFeats, int nTargets) {
        W = Nd4j.rand(nFeats, nTargets);
    }
}
