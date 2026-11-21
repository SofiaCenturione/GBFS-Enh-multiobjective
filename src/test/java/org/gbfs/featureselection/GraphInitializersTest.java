package org.gbfs.featureselection;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class GraphInitializersTest {
    private static final double[][] SIMILARITY = {
            {0.0, 0.9, 0.2},
            {0.9, 0.0, 0.5},
            {0.2, 0.5, 0.0}
    };

    @Test
    void buildsLocalThresholdGraphAndSymmetrizesNeighbors() {
        GraphInitializers.GraphInitResult result =
                GraphInitializers.localThreshold(SIMILARITY, 1, 0.8, true, true);

        assertEquals(0.9, result.adjacency()[0][1], 1e-12);
        assertEquals(0.9, result.adjacency()[1][0], 1e-12);
        assertArrayEquals(new int[]{1}, result.neighbors().get(0));
    }

    @Test
    void buildsFisherRedundancyDegrees() {
        GraphInitializers.GraphInitResult result =
                GraphInitializers.fisherRedundancyDegree(
                        SIMILARITY, new double[]{0.0, 0.5, 1.0}, 1, 2, true, false);

        assertEquals(1, result.degrees()[0]);
        assertEquals(2, result.degrees()[2]);
    }

    @Test
    void buildsMstAndAdditionalLocalEdges() {
        GraphInitializers.GraphInitResult result =
                GraphInitializers.mstPlusLocal(SIMILARITY, 0, true, true);

        assertEquals(2, result.neighbors().stream().mapToInt(row -> row.length).sum() / 2);
    }
}
