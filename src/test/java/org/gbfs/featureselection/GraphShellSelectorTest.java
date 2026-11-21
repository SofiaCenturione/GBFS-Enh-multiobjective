package org.gbfs.featureselection;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class GraphShellSelectorTest {
    @Test
    void keepsHighestWeightRepresentativeFromEachNonFinalShell() {
        double[][] adjacency = {
                {0.0, 1.0, 0.0},
                {1.0, 0.0, 1.0},
                {0.0, 1.0, 0.0}
        };
        double[] weights = {1.0, 3.0, 2.0};

        assertArrayEquals(new int[]{1, 2}, GraphShellSelector.select(adjacency, weights));
    }
}
