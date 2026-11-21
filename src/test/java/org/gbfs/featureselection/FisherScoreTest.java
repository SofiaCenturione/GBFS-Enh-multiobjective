package org.gbfs.featureselection;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class FisherScoreTest {
    @Test
    void scoresMulticlassFeaturesWithZeroWithinClassVariance() {
        double[][] features = {
                {0.0, 3.0},
                {0.0, 4.0},
                {10.0, 3.0},
                {10.0, 4.0}
        };
        double[] labels = {0.0, 0.0, 1.0, 1.0};

        assertArrayEquals(new double[]{100.0, 0.0}, FisherScore.score(features, labels), 1e-12);
    }
}
