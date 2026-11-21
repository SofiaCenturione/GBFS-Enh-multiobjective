package org.gbfs.featureselection;

import org.gbfs.math.MatrixOps;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public final class GraphShellSelector {
    private GraphShellSelector() {}

    public static int[] select(double[][] adjacency, double[] featureWeights) {
        MatrixOps.validateRectangular(adjacency);
        int nodeCount = adjacency.length;
        if (adjacency[0].length != nodeCount) {
            throw new IllegalArgumentException("Adjacency matrix must be square.");
        }
        if (featureWeights == null || featureWeights.length != nodeCount) {
            throw new IllegalArgumentException("There must be one feature weight per graph node.");
        }

        double[][] graph = copy(adjacency);
        List<Integer> remaining = new ArrayList<>(nodeCount);
        for (int node = 0; node < nodeCount; node++) {
            remaining.add(node);
        }

        List<int[]> shells = new ArrayList<>();
        while (!remaining.isEmpty()) {
            double[] degrees = weightedDegrees(graph, remaining, featureWeights);
            double minimum = Arrays.stream(degrees).min().orElseThrow();
            List<Integer> currentShell = new ArrayList<>();
            while (!remaining.isEmpty()) {
                degrees = weightedDegrees(graph, remaining, featureWeights);
                List<Integer> positions = new ArrayList<>();
                for (int position = 0; position < degrees.length; position++) {
                    if (Double.compare(degrees[position], minimum) == 0) {
                        positions.add(position);
                    }
                }
                if (positions.isEmpty()) {
                    break;
                }

                List<Integer> removedNodes = new ArrayList<>(positions.size());
                for (int position : positions) {
                    int node = remaining.get(position);
                    removedNodes.add(node);
                    currentShell.add(node);
                }
                for (int position = positions.size() - 1; position >= 0; position--) {
                    remaining.remove((int) positions.get(position));
                }
                for (int node : removedNodes) {
                    Arrays.fill(graph[node], 0.0);
                    for (double[] row : graph) {
                        row[node] = 0.0;
                    }
                }
            }
            if (!currentShell.isEmpty()) {
                shells.add(toIntArray(currentShell));
            }
        }

        int[] selected;
        if (shells.size() > 1) {
            List<Integer> representatives = new ArrayList<>(shells.size());
            for (int shellIndex = 0; shellIndex < shells.size() - 1; shellIndex++) {
                int[] shell = shells.get(shellIndex);
                if (shell.length == 0) {
                    continue;
                }
                int best = shell[0];
                for (int candidate : shell) {
                    if (featureWeights[candidate] > featureWeights[best]) {
                        best = candidate;
                    }
                }
                representatives.add(best);
            }
            int[] lastShell = shells.get(shells.size() - 1);
            selected = new int[representatives.size() + lastShell.length];
            for (int i = 0; i < representatives.size(); i++) {
                selected[i] = representatives.get(i);
            }
            System.arraycopy(lastShell, 0, selected, representatives.size(), lastShell.length);
        } else {
            selected = shells.get(0);
        }
        Arrays.sort(selected);
        return selected;
    }

    private static double[] weightedDegrees(double[][] adjacency, List<Integer> nodes, double[] weights) {
        double[] degrees = new double[nodes.size()];
        for (int position = 0; position < nodes.size(); position++) {
            int node = nodes.get(position);
            int neighborCount = 0;
            double edgeWeightSum = 0.0;
            for (double edgeWeight : adjacency[node]) {
                if (edgeWeight != 0.0) {
                    neighborCount++;
                    edgeWeightSum += edgeWeight;
                }
            }
            double product = neighborCount * edgeWeightSum * weights[node];
            if (product < 0.0 || !Double.isFinite(product)) {
                throw new IllegalArgumentException("Weighted graph degrees must be finite and non-negative.");
            }
            degrees[position] = Math.rint(Math.sqrt(product));
        }
        return degrees;
    }

    private static double[][] copy(double[][] values) {
        double[][] result = new double[values.length][];
        for (int row = 0; row < values.length; row++) {
            result[row] = values[row].clone();
        }
        return result;
    }

    private static int[] toIntArray(List<Integer> values) {
        return values.stream().mapToInt(Integer::intValue).toArray();
    }
}
