package org.gbfs.featureselection;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class Nsga3BinaryOptimizerTest {
    @Test
    void seededSearchIsRepeatableAndReturnsParetoSolutions() {
        Nsga3BinaryOptimizer.Evaluator evaluator = genome -> {
            int selected = 0;
            List<Integer> features = new ArrayList<>();
            for (int bit = 0; bit < genome.length; bit++) {
                if (genome[bit]) {
                    selected++;
                    features.add(bit);
                }
            }
            int[] mask = features.stream().mapToInt(Integer::intValue).toArray();
            return new Nsga3BinaryOptimizer.Evaluation(
                    new double[]{selected, 3 - selected, genome[0] ? 1 : 0,
                            genome[1] ? 1 : 0, genome[2] ? 1 : 0}, mask);
        };

        Nsga3BinaryOptimizer.Result first = Nsga3BinaryOptimizer.optimize(
                3, 3, 4, 2, 42, evaluator);
        Nsga3BinaryOptimizer.Result second = Nsga3BinaryOptimizer.optimize(
                3, 3, 4, 2, 42, evaluator);

        assertArrayEquals(first.bestGenome(), second.bestGenome());
        assertTrue(first.paretoMasks().length > 0);
        assertEquals(first.paretoMasks().length, first.paretoObjectives().length);
    }

    @Test
    void rejectsPopulationLargerThanGenomeSpace() {
        assertThrows(IllegalArgumentException.class,
                () -> Nsga3BinaryOptimizer.optimize(2, 2, 5, 1, 42,
                        genome -> new Nsga3BinaryOptimizer.Evaluation(
                                new double[]{0, 0, 0, 0, 0}, new int[0])));
    }
}
