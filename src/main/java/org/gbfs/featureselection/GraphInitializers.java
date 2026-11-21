package org.gbfs.featureselection;

import org.gbfs.math.MatrixOps;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

public final class GraphInitializers {
    public record GraphInitResult(double[][] adjacency, List<int[]> neighbors, int[] degrees) {}

    private GraphInitializers() {}

    public static GraphInitResult knn(double[][] similarity, int k) {
        validateSquare(similarity);
        int size = similarity.length;
        if (k < 0) {
            throw new IllegalArgumentException("k must not be negative.");
        }
        double[][] adjacency = new double[size][size];
        List<int[]> neighbors = new ArrayList<>(size);
        for (int node = 0; node < size; node++) {
            Integer[] order = new Integer[size];
            for (int candidate = 0; candidate < size; candidate++) {
                order[candidate] = candidate;
            }
            final int currentNode = node;
            Arrays.sort(order, Comparator.<Integer>comparingDouble(candidate ->
                    candidate == currentNode ? 0.0 : similarity[currentNode][candidate]).reversed());
            int count = Math.min(k, size);
            int[] selected = new int[count];
            for (int offset = 0; offset < count; offset++) {
                int candidate = order[offset];
                selected[offset] = candidate;
                adjacency[node][candidate] = similarity[node][candidate];
            }
            neighbors.add(selected);
        }
        return new GraphInitResult(adjacency, neighbors, null);
    }

    public static GraphInitResult localThreshold(double[][] similarity, int minimumNeighbors,
                                                 double quantile, boolean excludeSelf, boolean symmetrize) {
        validateSquare(similarity);
        validateNeighborCount(minimumNeighbors);
        if (quantile < 0.0 || quantile > 1.0) {
            throw new IllegalArgumentException("quantile must be between 0 and 1.");
        }
        int size = similarity.length;
        double[][] adjacency = new double[size][size];
        for (int node = 0; node < size; node++) {
            double[] positive = Arrays.stream(similarity[node]).filter(value -> value > 0.0).toArray();
            if (positive.length == 0) {
                continue;
            }
            Arrays.sort(positive);
            double threshold = quantile(positive, quantile);
            List<Integer> selected = new ArrayList<>();
            for (int candidate = 0; candidate < size; candidate++) {
                if ((!excludeSelf || candidate != node) && similarity[node][candidate] >= threshold) {
                    selected.add(candidate);
                }
            }
            if (selected.size() < minimumNeighbors) {
                Integer[] order = orderedNeighbors(similarity[node], node, excludeSelf);
                selected.clear();
                for (int offset = 0; offset < Math.min(minimumNeighbors, order.length); offset++) {
                    selected.add(order[offset]);
                }
            }
            for (int candidate : selected) {
                if (candidate != node) {
                    adjacency[node][candidate] = similarity[node][candidate];
                }
            }
        }
        if (symmetrize) {
            symmetrize(adjacency);
        }
        return resultFromNonZeroAdjacency(adjacency, null);
    }

    public static GraphInitResult fisherRedundancyDegree(double[][] similarity, double[] fisherScores,
                                                          int minimumNeighbors, int maximumNeighbors,
                                                          boolean excludeSelf, boolean symmetrize) {
        validateSquare(similarity);
        validateScores(fisherScores, similarity.length);
        validateNeighborCount(minimumNeighbors);
        int size = similarity.length;
        int upper = Math.min(maximumNeighbors, size - 1);
        double[] redundancy = new double[size];
        for (int node = 0; node < size; node++) {
            for (int candidate = 0; candidate < size; candidate++) {
                if (!excludeSelf || candidate != node) {
                    redundancy[node] += Math.abs(similarity[node][candidate]);
                }
            }
            redundancy[node] /= size;
        }
        double[] normalizedFisher = normalize(fisherScores);
        double[] normalizedRedundancy = normalize(redundancy);
        int[] degrees = new int[size];
        double[][] adjacency = new double[size][size];
        for (int node = 0; node < size; node++) {
            double score = 0.5 * normalizedFisher[node] + 0.5 * (1.0 - normalizedRedundancy[node]);
            int degree = (int) Math.rint(minimumNeighbors + (upper - minimumNeighbors) * score);
            degree = Math.max(minimumNeighbors, Math.min(degree, upper));
            degrees[node] = degree;
            Integer[] order = orderedNeighbors(similarity[node], node, excludeSelf);
            for (int offset = 0; offset < Math.min(degree, order.length); offset++) {
                int neighbor = order[offset];
                adjacency[node][neighbor] = similarity[node][neighbor];
            }
        }
        if (symmetrize) {
            symmetrize(adjacency);
        }
        return resultFromNonZeroAdjacency(adjacency, degrees);
    }

    public static GraphInitResult mstPlusLocal(double[][] similarity, int extraNeighbors,
                                                boolean excludeSelf, boolean symmetrize) {
        validateSquare(similarity);
        if (extraNeighbors < 0) {
            throw new IllegalArgumentException("extraNeighbors must not be negative.");
        }
        int size = similarity.length;
        double[][] adjacency = new double[size][size];
        List<Edge> edges = new ArrayList<>(size * Math.max(0, size - 1) / 2);
        for (int left = 0; left < size; left++) {
            for (int right = left + 1; right < size; right++) {
                edges.add(new Edge(left, right, Math.max(0.0, 1.0 - similarity[left][right])));
            }
        }
        edges.sort(Comparator.comparingDouble(Edge::distance)
                .thenComparingInt(Edge::left)
                .thenComparingInt(Edge::right));
        DisjointSet components = new DisjointSet(size);
        for (Edge edge : edges) {
            if (components.union(edge.left(), edge.right())) {
                adjacency[edge.left()][edge.right()] = similarity[edge.left()][edge.right()];
                adjacency[edge.right()][edge.left()] = similarity[edge.right()][edge.left()];
            }
        }

        for (int node = 0; node < size; node++) {
            double[] row = similarity[node].clone();
            if (excludeSelf) {
                row[node] = 0.0;
            }
            for (int candidate = 0; candidate < size; candidate++) {
                if (adjacency[node][candidate] > 0.0) {
                    row[candidate] = 0.0;
                }
            }
            Integer[] order = orderedNeighbors(row, node, excludeSelf);
            for (int offset = 0; offset < Math.min(extraNeighbors, order.length); offset++) {
                int neighbor = order[offset];
                if (neighbor != node) {
                    adjacency[node][neighbor] = similarity[node][neighbor];
                    adjacency[neighbor][node] = similarity[neighbor][node];
                }
            }
        }
        if (symmetrize) {
            symmetrize(adjacency);
        }
        return resultFromNonZeroAdjacency(adjacency, null);
    }

    public static GraphInitResult probabilistic(double[][] similarity, double[] fisherScores,
                                                 int minimumNeighbors, int maximumNeighbors, double beta,
                                                 boolean excludeSelf, boolean symmetrize, long seed) {
        validateSquare(similarity);
        validateScores(fisherScores, similarity.length);
        validateNeighborCount(minimumNeighbors);
        if (!Double.isFinite(beta) || beta < 0.0) {
            throw new IllegalArgumentException("beta must be finite and non-negative.");
        }
        int size = similarity.length;
        int upper = Math.min(maximumNeighbors, size - 1);
        double[] normalizedFisher = normalize(fisherScores);
        double[][] adjacency = new double[size][size];
        Random random = new Random(seed);
        for (int node = 0; node < size; node++) {
            double degreeEstimate = minimumNeighbors
                    + (upper - minimumNeighbors) * normalizedFisher[node];
            int degree = (int) Math.rint(degreeEstimate);
            degree = Math.max(minimumNeighbors, Math.min(degree, size - 1));

            List<Integer> candidates = new ArrayList<>();
            List<Double> weights = new ArrayList<>();
            for (int candidate = 0; candidate < size; candidate++) {
                if (excludeSelf && candidate == node) {
                    continue;
                }
                candidates.add(candidate);
                double sim = Math.max(similarity[node][candidate], 0.0);
                weights.add(Math.pow(sim, beta)
                        * 0.5 * (normalizedFisher[node] + normalizedFisher[candidate]));
            }

            int count = Math.min(degree, candidates.size());
            if (weights.stream().noneMatch(weight -> weight > 0.0)) {
                Integer[] order = orderedNeighbors(similarity[node], node, excludeSelf);
                for (int offset = 0; offset < count; offset++) {
                    int neighbor = order[offset];
                    adjacency[node][neighbor] = similarity[node][neighbor];
                }
                continue;
            }

            for (int selected = 0; selected < count; selected++) {
                double total = weights.stream().mapToDouble(Double::doubleValue).sum();
                double draw = random.nextDouble(total);
                int chosen = weights.size() - 1;
                for (int candidate = 0; candidate < weights.size(); candidate++) {
                    draw -= weights.get(candidate);
                    if (draw < 0.0) {
                        chosen = candidate;
                        break;
                    }
                }
                int neighbor = candidates.remove(chosen);
                weights.remove(chosen);
                adjacency[node][neighbor] = similarity[node][neighbor];
            }
        }
        if (symmetrize) {
            symmetrize(adjacency);
        }
        return resultFromNonZeroAdjacency(adjacency, null);
    }

    private static GraphInitResult resultFromNonZeroAdjacency(double[][] adjacency, int[] degrees) {
        List<int[]> neighbors = new ArrayList<>(adjacency.length);
        for (double[] row : adjacency) {
            int count = (int) Arrays.stream(row).filter(value -> value > 0.0).count();
            int[] rowNeighbors = new int[count];
            int position = 0;
            for (int candidate = 0; candidate < row.length; candidate++) {
                if (row[candidate] > 0.0) {
                    rowNeighbors[position++] = candidate;
                }
            }
            neighbors.add(rowNeighbors);
        }
        return new GraphInitResult(adjacency, neighbors, degrees);
    }

    private static Integer[] orderedNeighbors(double[] row, int node, boolean excludeSelf) {
        Integer[] order = new Integer[row.length];
        int count = 0;
        for (int candidate = 0; candidate < row.length; candidate++) {
            if (!excludeSelf || candidate != node) {
                order[count++] = candidate;
            }
        }
        order = Arrays.copyOf(order, count);
        Arrays.sort(order, Comparator.<Integer>comparingDouble(candidate -> row[candidate]).reversed());
        return order;
    }

    private static double[] normalize(double[] values) {
        double minimum = Arrays.stream(values).min().orElseThrow();
        double maximum = Arrays.stream(values).max().orElseThrow();
        double[] result = new double[values.length];
        if (maximum - minimum < 1e-12) {
            Arrays.fill(result, 0.5);
        } else {
            for (int i = 0; i < values.length; i++) {
                result[i] = (values[i] - minimum) / (maximum - minimum);
            }
        }
        return result;
    }

    private static double quantile(double[] sortedValues, double probability) {
        double position = probability * (sortedValues.length - 1);
        int lower = (int) Math.floor(position);
        int upper = (int) Math.ceil(position);
        double weight = position - lower;
        return sortedValues[lower] * (1.0 - weight) + sortedValues[upper] * weight;
    }

    private static void symmetrize(double[][] adjacency) {
        for (int left = 0; left < adjacency.length; left++) {
            for (int right = left + 1; right < adjacency.length; right++) {
                double value = Math.max(adjacency[left][right], adjacency[right][left]);
                adjacency[left][right] = value;
                adjacency[right][left] = value;
            }
        }
    }

    private static void validateSquare(double[][] matrix) {
        MatrixOps.validateRectangular(matrix);
        if (matrix.length != matrix[0].length) {
            throw new IllegalArgumentException("Similarity matrix must be square.");
        }
    }

    private static void validateScores(double[] scores, int expectedSize) {
        if (scores == null || scores.length != expectedSize) {
            throw new IllegalArgumentException("There must be one Fisher score per graph node.");
        }
    }

    private static void validateNeighborCount(int count) {
        if (count < 0) {
            throw new IllegalArgumentException("Neighbor count must not be negative.");
        }
    }

    private record Edge(int left, int right, double distance) {}

    private static final class DisjointSet {
        private final int[] parent;
        private final byte[] rank;

        private DisjointSet(int size) {
            parent = new int[size];
            rank = new byte[size];
            for (int i = 0; i < size; i++) {
                parent[i] = i;
            }
        }

        private int find(int value) {
            if (parent[value] != value) {
                parent[value] = find(parent[value]);
            }
            return parent[value];
        }

        private boolean union(int left, int right) {
            int leftRoot = find(left);
            int rightRoot = find(right);
            if (leftRoot == rightRoot) {
                return false;
            }
            if (rank[leftRoot] < rank[rightRoot]) {
                parent[leftRoot] = rightRoot;
            } else if (rank[leftRoot] > rank[rightRoot]) {
                parent[rightRoot] = leftRoot;
            } else {
                parent[rightRoot] = leftRoot;
                rank[leftRoot]++;
            }
            return true;
        }
    }
}
