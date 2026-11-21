package org.gbfs.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class MatrixOpsTest {
    @Test
    void scalesEachColumnAndMapsConstantColumnsToZero() {
        double[][] result = MatrixOps.minMaxZeroOne(new double[][]{
                {2.0, 5.0},
                {4.0, 5.0}
        });

        assertArrayEquals(new double[]{0.0, 0.0}, result[0]);
        assertArrayEquals(new double[]{1.0, 0.0}, result[1]);
    }

    @Test
    void computesAbsolutePearsonCorrelationAndUsesZeroForConstantColumns() {
        double[][] result = MatrixOps.absoluteCorrelation(new double[][]{
                {0.0, 3.0, 7.0},
                {1.0, 2.0, 7.0},
                {2.0, 1.0, 7.0}
        });

        assertEquals(1.0, result[0][1], 1e-12);
        assertEquals(0.0, result[0][2], 1e-12);
        assertEquals(0.0, result[2][2], 1e-12);
    }
}
