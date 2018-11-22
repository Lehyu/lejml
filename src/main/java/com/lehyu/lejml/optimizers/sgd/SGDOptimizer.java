/*
 * Copyright (C) 2018 Baidu, Inc. All Rights Reserved.
 */

// created by lianghongyu at 2018/10/25

package com.lehyu.lejml.optimizers.sgd;

import org.nd4j.linalg.api.ndarray.INDArray;

import com.lehyu.lejml.loss.ILoss;
import com.lehyu.lejml.loss.LossUtils;
import com.lehyu.lejml.loss.impl.LogLoss;
import com.lehyu.lejml.normalizer.L1Normalizer;
import com.lehyu.lejml.normalizer.L2Normalizer;
import com.lehyu.lejml.optimizers.IOptimizer;
import com.lehyu.lejml.utils.RandomUtils;

public class SGDOptimizer implements IOptimizer {
    private int maxIter; // max iteration
    private int earlyStopping; // early stopping
    private int batch; // batch
    private double eta; // learning rate
    private double lambda; // l2 penalty
    private double gamma; // l1 penalty
    private double eta0; // initial learning ratio
    private ILoss loss;
    private double iterLoss = Double.MAX_VALUE; // record the min loss of model
    private int notDecreaseIter = 0;

    public static class Builder {
        private int maxIter = 1000;
        private int earlyStopping = 5;
        private int batch = 20;
        private double eta = 0.1;
        private double lambda = 0.0001;
        private double gamma = 0;
        private ILoss loss = new LogLoss();

        public Builder eta(double eta) {
            this.eta = eta;
            return this;
        }

        public Builder maxIter(int maxIter) {
            this.maxIter = maxIter;
            return this;
        }

        public Builder earlyStopping(int earlyStopping) {
            this.earlyStopping = earlyStopping;
            return this;
        }

        public Builder lambda(double lambda) {
            this.lambda = lambda;
            return this;
        }

        public Builder gamma(double gamma) {
            this.gamma = gamma;
            return this;
        }

        public SGDOptimizer build() {
            return new SGDOptimizer(this);
        }

        public Builder loss(String name) {
            loss = LossUtils.getLossByName(name);
            return this;
        }

        public Builder batch(int batch) {
            this.batch = batch;
            return this;
        }
    }

    public SGDOptimizer() {
        this(new Builder());
    }

    SGDOptimizer(Builder builder) {
        this.maxIter = builder.maxIter;
        this.earlyStopping = builder.earlyStopping;
        this.eta = builder.eta;
        this.lambda = builder.lambda;
        this.gamma = builder.gamma;
        this.eta0 = eta;
        this.loss = builder.loss;
        this.batch = builder.batch;
    }

    public void setLoss(String name) {
        loss = LossUtils.getLossByName(name);
    }
    public INDArray optim(INDArray X, INDArray y, INDArray W) {
        int nSamples = X.rows();
        int nBatchs = getTotalBatch(nSamples);
        for (int iter = 0; iter < this.maxIter; iter++) {
            int[] indices = RandomUtils.shuffle(nSamples);
            X = X.getRows(indices);
            y = y.getRows(indices);
            double curLoss = 0;
            for (int nBatch = 0; nBatch < nBatchs; nBatch++) {
                int[] rows = getIndices(nBatch, nSamples);
                INDArray subX = X.getRows(rows);
                INDArray subY = y.getRows(rows);
                curLoss += loss.computeLoss(subX, subY, W);
                INDArray diffW = loss.derive(subX, subY, W);
                W = updateWeights(W, diffW);
            }
            curLoss /= nSamples;
            if (earlyStopping(curLoss)) {
                System.out.println("Early stopping at iteration: " + iter + ", loss: " + this.iterLoss);
                break;
            }
            updateEta(iter);
        }
        return W;
    }

    private int[] getIndices(int nBatch, int nSamples) {
        int startRow = getStart(nBatch);
        int endRow = getEnd(nBatch, nSamples);
        int[] indices = new int[endRow-startRow];
        for (int index = startRow; index < endRow; index++) {
            indices[index-startRow] = index;
        }
        return indices;
    }

    private boolean earlyStopping(double curLoss) {
        if (curLoss < this.iterLoss) {
            this.iterLoss = curLoss;
            this.notDecreaseIter = 0;
        } else if (curLoss >= this.iterLoss && this.notDecreaseIter < this.earlyStopping) {
            this.notDecreaseIter += 1;
        } else {
            return true;
        }
        return false;
    }

    private void updateEta(int iter) {
    }

    private INDArray updateWeights(INDArray W, INDArray diffW) {
        if (0 != this.lambda) {
            diffW = diffW.add(W.mul(this.lambda));
        }
        if (0 != this.gamma) {
            diffW = diffW.add(L1Normalizer.norm(W).mul(this.gamma));
        }
        W = W.sub(diffW.mul(this.eta));
        return W;
    }

    private int getStart(int nBatch) {
        return nBatch * this.batch;
    }

    private int getTotalBatch(int nSamples) {
        return nSamples / batch + (nSamples % batch == 0 ? 0 : 1);
    }

    private int getEnd(int nBatch, int nSamples) {
        int end = (nBatch + 1) * this.batch;
        return end >= nSamples ? nSamples : end;
    }
}
