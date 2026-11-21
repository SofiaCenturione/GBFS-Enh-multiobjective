package org.gbfs.featureselection;

import org.gbfs.math.MatrixOps;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public final class EnhancedFeatureSelector {
    public record Selection(int[] core, int[] postNormal, int[] postBuddy, int[][] pareto,
                            double[][] paretoObjectives, boolean[] neighborMask,
                            double solverSeconds, double buddySeconds) {}

    private EnhancedFeatureSelector() {}

    public static Selection select(String initializer, String kshellMode, double[][] train,
                                   double[] labels, int neighbors, int folds, int seed,
                                   int maxAdd, double rcTau, double buddyLambda, double beta,
                                   int population, int generations) throws Exception {
        MatrixOps.validateRectangular(train);
        if (labels == null || labels.length != train.length || train[0].length < 1) {
            throw new IllegalArgumentException("Training data and labels must be non-empty and aligned.");
        }
        double[] fisher = FisherScore.score(train, labels);
        double[] weights = normalizeWeights(fisher);
        double[][] similarity = MatrixOps.absoluteCorrelation(train);
        for (int feature = 0; feature < similarity.length; feature++) {
            similarity[feature][feature] = 0.0;
        }

        GraphInitializers.GraphInitResult graph = initialize(initializer, similarity, fisher,
                neighbors, beta, seed);
        double[][] complementarity = complementarity(
                train, labels, graph.adjacency(), folds, seed, neighbors);
        ObjectiveData objectiveData = new ObjectiveData(train, labels, seed);
        int variableCount = train[0].length
                + graph.neighbors().stream().mapToInt(row -> row.length).sum();

        long solverStarted = System.nanoTime();
        Nsga3BinaryOptimizer.Result optimized = Nsga3BinaryOptimizer.optimize(
                variableCount, train[0].length, population, generations, 42,
                genome -> evaluateGenome(genome, train[0].length, graph.adjacency(), weights,
                        complementarity, similarity, objectiveData, kshellMode, maxAdd, rcTau));
        double solverSeconds = (System.nanoTime() - solverStarted) / 1e9;
        int[] best = selected(optimized.bestGenome(), train[0].length);
        int[] normal = best.clone();
        long buddyStarted = System.nanoTime();
        int[] buddy = buddy(train, labels, best, complementarity, similarity,
                neighbors, folds, seed, buddyLambda);
        double buddySeconds = (System.nanoTime() - buddyStarted) / 1e9;
        boolean[] bestGenome = optimized.bestGenome();
        boolean[] neighborMask = Arrays.copyOfRange(bestGenome, train[0].length, bestGenome.length);
        return new Selection(best, normal, buddy, optimized.paretoMasks(),
                optimized.paretoObjectives(), neighborMask, solverSeconds, buddySeconds);
    }

    private static Nsga3BinaryOptimizer.Evaluation evaluateGenome(
            boolean[] genome, int featureCount, double[][] adjacency, double[] weights,
            double[][] complementarity, double[][] similarity, ObjectiveData objectiveData,
            String kshellMode, int maxAdd, double rcTau) {
        double[][] masked = new double[featureCount][featureCount];
        boolean hasEdge = false;
        for (int left = 0; left < featureCount; left++) {
            if (!genome[left]) continue;
            for (int right = 0; right < featureCount; right++) {
                if (genome[right]) {
                    masked[left][right] = adjacency[left][right];
                    hasEdge |= masked[left][right] != 0.0;
                }
            }
        }
        if (!hasEdge) {
            return new Nsga3BinaryOptimizer.Evaluation(
                    new double[]{Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY,
                            Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY},
                    range(featureCount));
        }

        int[] core = GraphShellSelector.select(masked, weights);
        int[] extended = switch (kshellMode) {
            case "normal" -> core;
            case "rc_greedy" -> rcGreedy(core, complementarity, similarity, maxAdd, rcTau);
            default -> throw new IllegalArgumentException("Unsupported k-shell mode: " + kshellMode);
        };
        double[] coreObjectives = objectiveData.evaluate(core);
        double[] extendedObjectives = Arrays.equals(core, extended)
                ? coreObjectives : objectiveData.evaluate(extended);
        int[] selected = dominates(coreObjectives, extendedObjectives)
                || (!dominates(extendedObjectives, coreObjectives)
                && norm(coreObjectives) < norm(extendedObjectives)) ? core : extended;
        double[] objectives = selected == core ? coreObjectives : extendedObjectives;
        return new Nsga3BinaryOptimizer.Evaluation(objectives, selected);
    }

    private static boolean dominates(double[] left, double[] right) {
        boolean strictlyBetter = false;
        for (int i = 0; i < left.length; i++) {
            if (left[i] > right[i]) return false;
            if (left[i] < right[i]) strictlyBetter = true;
        }
        return strictlyBetter;
    }

    private static double norm(double[] values) {
        double sum = 0.0;
        for (double value : values) sum += value * value;
        return Math.sqrt(sum);
    }

    private static GraphInitializers.GraphInitResult initialize(String name, double[][] similarity,
                                                                 double[] fisher, int k, double beta,
                                                                 int seed) {
        int featureCount = similarity.length;
        int maximum = Math.min(Math.max(k, 1) * 3, featureCount - 1);
        return switch (name) {
            case "knn" -> GraphInitializers.knn(similarity, Math.min(k, featureCount - 1));
            case "local_thresh" -> GraphInitializers.localThreshold(similarity, 1, 0.8, true, true);
            case "fisher_red_deg" -> GraphInitializers.fisherRedundancyDegree(
                    similarity, fisher, 1, maximum, true, true);
            case "mst_plus_local" -> GraphInitializers.mstPlusLocal(similarity, 2, true, true);
            case "probabilistic" -> GraphInitializers.probabilistic(
                    similarity, fisher, 1, maximum, beta, true, true, seed);
            default -> throw new IllegalArgumentException("Unsupported initializer: " + name);
        };
    }

    private static double[] normalizeWeights(double[] scores) {
        double minimum = Arrays.stream(scores).min().orElse(0.0);
        double maximum = Arrays.stream(scores).max().orElse(0.0);
        double[] result = new double[scores.length];
        for (int i = 0; i < scores.length; i++) {
            result[i] = maximum == minimum ? 1.0 : 1.0 + (scores[i] - minimum) / (maximum - minimum);
        }
        return result;
    }

    private static double[][] complementarity(double[][] x, double[] y, double[][] adjacency,
                                              int folds, int seed, int k) throws Exception {
        int features = x[0].length;
        double[] singles = new double[features];
        for (int feature = 0; feature < features; feature++) {
            singles[feature] = crossValidation(x, y, new int[]{feature}, folds, seed, k);
        }
        double[][] result = new double[features][features];
        for (int left = 0; left < features; left++) {
            for (int right = left + 1; right < features; right++) {
                if (adjacency[left][right] == 0.0 && adjacency[right][left] == 0.0) continue;
                double accuracy = crossValidation(x, y, new int[]{left, right}, folds, seed, k);
                result[left][right] = result[right][left] =
                        Math.max(0.0, accuracy - Math.max(singles[left], singles[right]));
            }
        }
        return result;
    }

    private static int[] rcGreedy(int[] initial, double[][] complementarity, double[][] similarity,
                                  int maxAdd, double tau) {
        List<Integer> selected = toList(initial);
        for (int addition = 0; addition < maxAdd; addition++) {
            int best = -1;
            double bestScore = -1e9;
            for (int feature = 0; feature < similarity.length; feature++) {
                if (selected.contains(feature)) continue;
                double comp = 0.0;
                double red = 0.0;
                for (int existing : selected) {
                    comp = Math.max(comp, complementarity[feature][existing]);
                    red = Math.max(red, Math.abs(1.0 - similarity[feature][existing]));
                }
                double score = comp - tau * red;
                if (score > bestScore) {
                    bestScore = score;
                    best = feature;
                }
            }
            if (best < 0 || bestScore <= 0.0) break;
            selected.add(best);
        }
        return selected.stream().mapToInt(Integer::intValue).sorted().toArray();
    }

    private static int[] buddy(double[][] x, double[] y, int[] initial, double[][] complementarity,
                               double[][] similarity, int k, int folds, int seed, double lambda)
            throws Exception {
        List<Integer> selected = toList(initial);
        double currentAccuracy = selected.isEmpty() ? 0.0 : crossValidation(
                x, y, initial, folds, seed, k);

        for (int core : initial) {
            List<Integer> candidates = new ArrayList<>();
            for (int feature = 0; feature < similarity.length; feature++) {
                if (!selected.contains(feature)) candidates.add(feature);
            }
            candidates.sort((left, right) -> {
                double leftScore = complementarity[core][left] - lambda * Math.abs(1.0 - similarity[core][left]);
                double rightScore = complementarity[core][right] - lambda * Math.abs(1.0 - similarity[core][right]);
                int comparison = Double.compare(rightScore, leftScore);
                return comparison != 0 ? comparison : Integer.compare(left, right);
            });
            for (int candidate : candidates) {
                List<Integer> proposed = new ArrayList<>(selected);
                proposed.add(candidate);
                double accuracy = crossValidation(x, y,
                        proposed.stream().mapToInt(Integer::intValue).toArray(), folds, seed, k);
                if (accuracy > currentAccuracy) {
                    selected.add(candidate);
                    currentAccuracy = accuracy;
                    break;
                }
            }
        }
        return selected.stream().mapToInt(Integer::intValue).sorted().toArray();
    }

    private static double crossValidation(double[][] x, double[] labels, int[] subset,
                                          int folds, int seed, int k) throws Exception {
        int[][] validationFolds = stratifiedFolds(labels, folds, seed);
        double totalAccuracy = 0.0;
        for (int[] validation : validationFolds) {
            boolean[] isValidation = new boolean[labels.length];
            for (int index : validation) isValidation[index] = true;
            List<Integer> training = new ArrayList<>();
            for (int i = 0; i < labels.length; i++) if (!isValidation[i]) training.add(i);
            int[] trainIndices = training.stream().mapToInt(Integer::intValue).toArray();
            double[][] trainX = selectRows(x, trainIndices);
            double[] trainY = labelsAt(labels, trainIndices);
            double[][] validationX = selectRows(x, validation);
            double[] validationY = labelsAt(labels, validation);
            totalAccuracy += WekaKnnClassifier.accuracy(
                    trainX, trainY, validationX, validationY, subset, k);
        }
        return validationFolds.length == 0 ? 0.0 : totalAccuracy / validationFolds.length;
    }

    private static double[][] selectRows(double[][] values, int[] indices) {
        double[][] selected = new double[indices.length][];
        for (int row = 0; row < indices.length; row++) selected[row] = values[indices[row]];
        return selected;
    }

    private static double[] labelsAt(double[] labels, int[] indices) {
        double[] selected = new double[indices.length];
        for (int row = 0; row < indices.length; row++) selected[row] = labels[indices[row]];
        return selected;
    }

    private static int[][] stratifiedFolds(double[] labels, int folds, int seed) {
        java.util.Map<Double, List<Integer>> byClass = new java.util.TreeMap<>();
        for (int i = 0; i < labels.length; i++) byClass.computeIfAbsent(labels[i], ignored -> new ArrayList<>()).add(i);
        List<List<Integer>> result = new ArrayList<>();
        for (int fold = 0; fold < folds; fold++) result.add(new ArrayList<>());
        Random random = new Random(seed);
        for (List<Integer> indices : byClass.values()) {
            if (indices.size() < folds) throw new IllegalArgumentException("Each class must have at least cv_folds samples.");
            java.util.Collections.shuffle(indices, random);
            for (int i = 0; i < indices.size(); i++) result.get(i % folds).add(indices.get(i));
        }
        return result.stream().map(rows -> rows.stream().mapToInt(Integer::intValue).toArray()).toArray(int[][]::new);
    }

    private static int[] selected(boolean[] genome, int featureCount) {
        int count = 0;
        for (int i = 0; i < featureCount; i++) if (genome[i]) count++;
        int[] result = new int[count];
        for (int i = 0, index = 0; i < featureCount; i++) if (genome[i]) result[index++] = i;
        return result;
    }

    private static List<Integer> toList(int[] values) {
        List<Integer> result = new ArrayList<>(values.length);
        for (int value : values) result.add(value);
        return result;
    }

    private static int[] range(int size) {
        int[] result = new int[size];
        for (int i = 0; i < size; i++) result[i] = i;
        return result;
    }

    private static final class ObjectiveData {
        private final double[][] features;
        private final double[] labels;
        private final double[] relevance;
        private final double[][] redundancy;
        private final double[][] synergy;
        private final Random bootstrapRandom;

        private ObjectiveData(double[][] features, double[] labels, int seed) {
            this.features = features;
            this.labels = labels;
            this.bootstrapRandom = new Random(seed);
            int count = features[0].length;
            this.relevance = new double[count];
            this.redundancy = new double[count][count];
            this.synergy = new double[count][count];
            double labelEntropy = entropy(labels, 20);
            double[] featureEntropy = new double[count];
            for (int feature = 0; feature < count; feature++) {
                double[] values = column(features, feature);
                featureEntropy[feature] = entropy(values, 20);
                double mutualInformation = mutualInformation(values, labels, 20, 20);
                relevance[feature] = featureEntropy[feature] * labelEntropy == 0.0
                        ? 0.0 : mutualInformation / Math.sqrt(featureEntropy[feature] * labelEntropy);
            }
            for (int left = 0; left < count; left++) for (int right = left + 1; right < count; right++) {
                double[] leftValues = column(features, left);
                double[] rightValues = column(features, right);
                double mutualInformation = mutualInformation(leftValues, rightValues, 20, 20);
                redundancy[left][right] = redundancy[right][left] = mutualInformation;
                double jointInformation = jointMutualInformation(leftValues, rightValues, labels, 10);
                double individualInformation = relevance[left] * Math.sqrt(featureEntropy[left] * labelEntropy);
                double pairSynergy = mutualInformation * Math.max(0.0, jointInformation - individualInformation);
                synergy[left][right] = synergy[right][left] = pairSynergy;
            }
        }

        private double[] evaluate(int[] subset) {
            if (subset.length == 0) return new double[]{
                    Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY,
                    Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY};
            double meanRelevance = 0.0;
            for (int feature : subset) meanRelevance += relevance[feature];
            meanRelevance /= subset.length;
            double meanRedundancy = 0.0, meanSynergy = 0.0;
            int pairs = subset.length * (subset.length - 1) / 2;
            for (int left = 0; left < subset.length; left++) for (int right = left + 1; right < subset.length; right++) {
                meanRedundancy += redundancy[subset[left]][subset[right]];
                meanSynergy += synergy[subset[left]][subset[right]];
            }
            if (pairs > 0) {
                meanRedundancy /= pairs;
                meanSynergy /= pairs;
            }
            return new double[]{-meanRelevance, meanRedundancy, -meanSynergy,
                    (double) subset.length / relevance.length, -stability(subset)};
        }

        private double stability(int[] target) {
            boolean[] targetSet = new boolean[relevance.length];
            for (int feature : target) targetSet[feature] = true;
            double total = 0.0;
            for (int bootstrap = 0; bootstrap < 5; bootstrap++) {
                double[][] sample = new double[features.length][features[0].length];
                double[] sampleLabels = new double[labels.length];
                for (int row = 0; row < sample.length; row++) {
                    int selected = bootstrapRandom.nextInt(features.length);
                    sample[row] = features[selected].clone();
                    sampleLabels[row] = labels[selected];
                }
                double[] score = new double[relevance.length];
                for (int feature = 0; feature < score.length; feature++) {
                    score[feature] = mutualInformation(column(sample, feature), sampleLabels, 20, 20);
                }
                List<Integer> chosen = new ArrayList<>();
                boolean[] available = new boolean[score.length];
                Arrays.fill(available, true);
                for (int pick = 0; pick < target.length; pick++) {
                    int best = -1;
                    double bestScore = Double.NEGATIVE_INFINITY;
                    for (int candidate = 0; candidate < score.length; candidate++) {
                        if (!available[candidate]) continue;
                        double value = score[candidate];
                        if (!chosen.isEmpty()) {
                            double sum = 0.0;
                            for (int existing : chosen) {
                                sum += mutualInformation(column(sample, candidate), column(sample, existing), 20, 20);
                            }
                            value -= sum / chosen.size();
                        }
                        if (value > bestScore) {
                            bestScore = value;
                            best = candidate;
                        }
                    }
                    if (best < 0) break;
                    chosen.add(best);
                    available[best] = false;
                }
                int intersection = 0;
                boolean[] chosenSet = new boolean[targetSet.length];
                for (int feature : chosen) {
                    chosenSet[feature] = true;
                    if (targetSet[feature]) intersection++;
                }
                int union = 0;
                for (int feature = 0; feature < targetSet.length; feature++) {
                    if (targetSet[feature] || chosenSet[feature]) union++;
                }
                total += union == 0 ? 0.0 : (double) intersection / union;
            }
            return total / 5.0;
        }
    }

    private static double entropy(double[] values, int bins) {
        double minimum = Arrays.stream(values).min().orElse(0.0);
        double maximum = Arrays.stream(values).max().orElse(0.0);
        int[] counts = histogram(values, bins, minimum, maximum);
        double result = 0.0;
        for (int count : counts) if (count > 0) {
            double probability = (double) count / values.length;
            result -= probability * Math.log(probability);
        }
        return result;
    }

    private static double mutualInformation(double[] left, double[] right, int binsLeft, int binsRight) {
        int[][] joint = new int[binsLeft][binsRight];
        int[] leftCounts = new int[binsLeft], rightCounts = new int[binsRight];
        double leftMinimum = Arrays.stream(left).min().orElse(0.0);
        double leftMaximum = Arrays.stream(left).max().orElse(0.0);
        double rightMinimum = Arrays.stream(right).min().orElse(0.0);
        double rightMaximum = Arrays.stream(right).max().orElse(0.0);
        for (int row = 0; row < left.length; row++) {
            int a = bin(left[row], leftMinimum, leftMaximum, binsLeft);
            int b = bin(right[row], rightMinimum, rightMaximum, binsRight);
            joint[a][b]++;
            leftCounts[a]++;
            rightCounts[b]++;
        }
        double result = 0.0;
        for (int a = 0; a < binsLeft; a++) for (int b = 0; b < binsRight; b++) {
            int count = joint[a][b];
            if (count == 0) continue;
            double pxy = (double) count / left.length;
            double px = (double) leftCounts[a] / left.length;
            double py = (double) rightCounts[b] / left.length;
            result += pxy * Math.log(pxy / (px * py));
        }
        return result;
    }

    private static double jointMutualInformation(double[] first, double[] second, double[] labels, int bins) {
        int[][][] counts = new int[bins][bins][bins];
        int[][] firstSecond = new int[bins][bins];
        int[] labelCounts = new int[bins];
        double firstMinimum = Arrays.stream(first).min().orElse(0.0);
        double firstMaximum = Arrays.stream(first).max().orElse(0.0);
        double secondMinimum = Arrays.stream(second).min().orElse(0.0);
        double secondMaximum = Arrays.stream(second).max().orElse(0.0);
        double labelMinimum = Arrays.stream(labels).min().orElse(0.0);
        double labelMaximum = Arrays.stream(labels).max().orElse(0.0);
        for (int row = 0; row < labels.length; row++) {
            int a = bin(first[row], firstMinimum, firstMaximum, bins);
            int b = bin(second[row], secondMinimum, secondMaximum, bins);
            int c = bin(labels[row], labelMinimum, labelMaximum, bins);
            counts[a][b][c]++;
            firstSecond[a][b]++;
            labelCounts[c]++;
        }
        double result = 0.0;
        for (int a = 0; a < bins; a++) for (int b = 0; b < bins; b++) for (int c = 0; c < bins; c++) {
            int count = counts[a][b][c];
            if (count == 0) continue;
            double pxyz = (double) count / labels.length;
            double pxPair = (double) firstSecond[a][b] / labels.length;
            double py = (double) labelCounts[c] / labels.length;
            result += pxyz * Math.log(pxyz / (pxPair * py));
        }
        return result;
    }

    private static int[] histogram(double[] values, int bins, double minimum, double maximum) {
        int[] counts = new int[bins];
        for (double value : values) counts[bin(value, minimum, maximum, bins)]++;
        return counts;
    }

    private static int bin(double value, double minimum, double maximum, int bins) {
        if (maximum == minimum) return 0;
        return Math.max(0, Math.min(bins - 1, (int) ((value - minimum) / (maximum - minimum) * bins)));
    }

    private static double[] column(double[][] values, int feature) {
        double[] result = new double[values.length];
        for (int row = 0; row < values.length; row++) result[row] = values[row][feature];
        return result;
    }
}
