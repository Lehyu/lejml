/*
 * Copyright (C) 2018 Baidu, Inc. All Rights Reserved.
 */

// created by lianghongyu at 2018/10/29

package com.lehyu.lejml.models.linear_model;

import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.api.rng.distribution.impl.NormalDistribution;
import org.nd4j.linalg.api.rng.distribution.impl.UniformDistribution;
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
    protected boolean copied;

    public static class Builder<T extends Builder<T>> {
        private int nJobs = 1;
        private boolean normalize = true;
        private boolean fitIntercept = true;
        private boolean copied = false;
        private IOptimizer optimizer = new SGDOptimizer();

        public T nJobs(int nJobs) {
            this.nJobs = nJobs;
            return (T) this;
        }

        public T copied(boolean copied) {
            this.copied = copied;
            return (T) this;
        }

        public T normalize(boolean normalize) {
            this.normalize = normalize;
            return (T) this;
        }

        public T fitIntercept(boolean fitIntercept) {
            this.fitIntercept = fitIntercept;
            return (T) this;
        }

        public T optimizer(IOptimizer optimizer) {
            this.optimizer = optimizer;
            return (T) this;
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
        return this.fitIntercept ? Nd4j.append(X, 1, 1, -1) : X;
    }

    protected INDArray copy(INDArray X) {
        if (this.copied) {
            INDArray XCopy = Nd4j.zeros(X.rows(), X.columns());
            Nd4j.copy(X, XCopy);
            return XCopy;
        } else {
            return X;
        }
    }

    protected void initWeights(int nFeats, int nTargets) {
        Distribution dist = new UniformDistribution(-1,1);
        int[] shape = {nFeats, nTargets};
        W = Nd4j.rand(shape, dist);
    }
}
