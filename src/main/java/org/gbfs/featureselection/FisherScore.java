package org.gbfs.featureselection;

import org.gbfs.math.MatrixOps;

import java.util.Arrays;
import java.util.TreeSet;

public final class FisherScore {
    private FisherScore() {}

    public static double[] score(double[][] samplesByFeature, double[] labels) {
        MatrixOps.validateRectangular(samplesByFeature);
        if (labels == null || labels.length != samplesByFeature.length) {
            throw new IllegalArgumentException("Labels must have one value per sample.");
        }

        TreeSet<Double> classes = new TreeSet<>();
        for (double label : labels) {
            if (!Double.isFinite(label)) {
                throw new IllegalArgumentException("Labels must be finite.");
            }
            classes.add(label);
        }
        if (classes.isEmpty()) {
            throw new IllegalArgumentException("At least one class is required.");
        }

        int featureCount = samplesByFeature[0].length;
        double[] scores = new double[featureCount];
        for (int feature = 0; feature < featureCount; feature++) {
            double overallMean = 0.0;
            for (double[] sample : samplesByFeature) {
                overallMean += sample[feature];
            }
            overallMean /= samplesByFeature.length;

            double betweenClass = 0.0;
            double withinClass = 0.0;
            for (double classLabel : classes) {
                int count = 0;
                double classMean = 0.0;
                for (int row = 0; row < labels.length; row++) {
                    if (labels[row] == classLabel) {
                        count++;
                        classMean += samplesByFeature[row][feature];
                    }
                }
                classMean /= count;

                double classVariance = 0.0;
                for (int row = 0; row < labels.length; row++) {
                    if (labels[row] == classLabel) {
                        double difference = samplesByFeature[row][feature] - classMean;
                        classVariance += difference * difference;
                    }
                }
                classVariance /= count;

                double meanDifference = classMean - overallMean;
                betweenClass += count * meanDifference * meanDifference;
                withinClass += count * classVariance;
            }

            scores[feature] = betweenClass == 0.0
                    ? 0.0
                    : withinClass == 0.0 ? 100.0 : betweenClass / withinClass;
        }
        return scores;
    }

    public static int[] featureList(int featureCount) {
        if (featureCount < 0) {
            throw new IllegalArgumentException("Feature count must not be negative.");
        }
        int[] features = new int[featureCount];
        Arrays.setAll(features, index -> index + 1);
        return features;
    }
}
