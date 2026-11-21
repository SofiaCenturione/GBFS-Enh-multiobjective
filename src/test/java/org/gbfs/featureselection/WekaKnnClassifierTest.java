package org.gbfs.featureselection;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class WekaKnnClassifierTest {
    @Test
    void classifiesUsingSelectedFeaturesAndEuclideanKnn() throws Exception {
        double[][] train = {
                {0.0, 100.0},
                {0.2, 0.0},
                {9.8, 100.0},
                {10.0, 0.0}
        };
        double[] trainLabels = {0, 0, 1, 1};
        double[][] test = {
                {0.1, -500.0},
                {9.9, 500.0}
        };
        double[] testLabels = {0, 1};

        assertEquals(1.0, WekaKnnClassifier.accuracy(
                train, trainLabels, test, testLabels, new int[]{0}, 1));
    }

    @Test
    void emptyFeatureSetMatchesPythonRunnerAccuracyFallback() throws Exception {
        double[][] train = {{0.0}, {1.0}, {2.0}};
        double[] trainLabels = {1, 0, 1};
        double[][] test = {{3.0}, {4.0}};
        double[] testLabels = {1, 0};

        assertEquals(0.0, WekaKnnClassifier.accuracy(
                train, trainLabels, test, testLabels, new int[0], 3));
    }
}
