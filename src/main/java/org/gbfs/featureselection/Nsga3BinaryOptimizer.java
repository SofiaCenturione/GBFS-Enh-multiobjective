package org.gbfs.featureselection;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

public final class Nsga3BinaryOptimizer {
    @FunctionalInterface
    public interface Evaluator {
        Evaluation evaluate(boolean[] genome);
    }

    public record Evaluation(double[] objectives, int[] selectedFeatures) {}
    public record Result(boolean[] bestGenome, int[][] paretoMasks, double[][] paretoObjectives) {}

    private record Individual(boolean[] genome, Evaluation evaluation) {}
    private record Association(int direction, double distance) {}

    private Nsga3BinaryOptimizer() {}

    public static Result optimize(int variables, int featureCount, int populationSize, int generations,
                                  long seed, Evaluator evaluator) {
        if (variables < 1 || featureCount < 1 || featureCount > variables
                || populationSize < 2 || generations < 1
                || (variables < 31 && populationSize > (1L << variables))) {
            throw new IllegalArgumentException("Invalid NSGA-III dimensions or run parameters.");
        }
        Random random = new Random(seed);
        double[][] referenceDirections = energyDirections(populationSize, 5, seed ^ 0x4E534741334CL);
        List<Individual> population = randomPopulation(
                variables, populationSize, random, evaluator);

        for (int generation = 0; generation < generations; generation++) {
            List<Individual> offspring = breed(
                    population, populationSize, random, evaluator);
            List<Individual> combined = new ArrayList<>(population.size() + offspring.size());
            combined.addAll(population);
            combined.addAll(offspring);
            population = survival(combined, populationSize, referenceDirections, random);
        }

        List<Integer> firstFront = nondominatedFront(population);
        int best = closestToIdeal(population);
        int[][] paretoMasks = new int[firstFront.size()][];
        double[][] objectives = new double[firstFront.size()][];
        for (int index = 0; index < firstFront.size(); index++) {
            Individual individual = population.get(firstFront.get(index));
            paretoMasks[index] = selected(individual.genome(), featureCount);
            objectives[index] = individual.evaluation().objectives().clone();
        }
        return new Result(population.get(best).genome().clone(), paretoMasks, objectives);
    }

    private static List<Individual> randomPopulation(int variables, int size, Random random, Evaluator evaluator) {
        List<Individual> result = new ArrayList<>(size);
        Set<String> seen = new HashSet<>();
        while (result.size() < size) {
            boolean[] genome = new boolean[variables];
            for (int bit = 0; bit < variables; bit++) genome[bit] = random.nextBoolean();
            if (seen.add(key(genome))) result.add(new Individual(genome, evaluator.evaluate(genome)));
        }
        return result;
    }

    private static List<Individual> breed(List<Individual> population, int size,
                                          Random random, Evaluator evaluator) {
        int variables = population.getFirst().genome().length;
        List<Individual> children = new ArrayList<>(size);
        Set<String> seen = new HashSet<>();
        int attempts = 0;
        while (children.size() < size && attempts++ < size * 100) {
            Individual parentA = population.get(random.nextInt(population.size()));
            Individual parentB = population.get(random.nextInt(population.size()));
            boolean[] first = parentA.genome().clone();
            boolean[] second = parentB.genome().clone();
            if (variables > 2 && random.nextDouble() < 0.9) {
                int left = random.nextInt(variables - 1);
                int right = left + 1 + random.nextInt(variables - left - 1);
                for (int bit = left; bit < right; bit++) {
                    boolean value = first[bit];
                    first[bit] = second[bit];
                    second[bit] = value;
                }
            }
            mutate(first, random);
            mutate(second, random);
            addChild(first, evaluator, children, seen);
            if (children.size() < size) addChild(second, evaluator, children, seen);
        }
        while (children.size() < size) {
            boolean[] genome = new boolean[variables];
            for (int bit = 0; bit < variables; bit++) genome[bit] = random.nextBoolean();
            String key = key(genome);
            if (seen.add(key)) children.add(new Individual(genome, evaluator.evaluate(genome)));
        }
        return children;
    }

    private static void mutate(boolean[] genome, Random random) {
        double probability = 1.0 / genome.length;
        for (int bit = 0; bit < genome.length; bit++) {
            if (random.nextDouble() < probability) genome[bit] = !genome[bit];
        }
    }

    private static void addChild(boolean[] genome, Evaluator evaluator,
                                 List<Individual> children, Set<String> seen) {
        if (seen.add(key(genome))) children.add(new Individual(genome, evaluator.evaluate(genome)));
    }

    private static List<Individual> survival(List<Individual> candidates, int size,
                                             double[][] referenceDirections, Random random) {
        List<List<Integer>> fronts = nondominatedSort(candidates);
        List<Individual> selected = new ArrayList<>(size);
        List<Integer> lastFront = List.of();
        for (List<Integer> front : fronts) {
            if (selected.size() + front.size() <= size) {
                for (int index : front) selected.add(candidates.get(index));
            } else {
                lastFront = front;
                break;
            }
        }
        if (selected.size() == size) return selected;

        List<Individual> pool = new ArrayList<>(selected);
        for (int index : lastFront) pool.add(candidates.get(index));
        double[][] normalized = normalize(pool);
        Association[] associations = associate(normalized, referenceDirections);
        int[] nicheCounts = new int[referenceDirections.length];
        for (int i = 0; i < selected.size(); i++) nicheCounts[associations[i].direction()]++;

        List<Integer> remaining = new ArrayList<>();
        for (int i = selected.size(); i < pool.size(); i++) remaining.add(i);
        while (selected.size() < size && !remaining.isEmpty()) {
            int minimum = Integer.MAX_VALUE;
            for (int index : remaining) minimum = Math.min(minimum, nicheCounts[associations[index].direction()]);
            List<Integer> eligibleDirections = new ArrayList<>();
            for (int direction = 0; direction < nicheCounts.length; direction++) {
                int candidateDirection = direction;
                if (nicheCounts[direction] == minimum
                        && remaining.stream().anyMatch(index -> associations[index].direction() == candidateDirection)) {
                    eligibleDirections.add(direction);
                }
            }
            int direction = eligibleDirections.get(random.nextInt(eligibleDirections.size()));
            List<Integer> associated = new ArrayList<>();
            for (int index : remaining) if (associations[index].direction() == direction) associated.add(index);
            int chosen;
            if (nicheCounts[direction] == 0) {
                chosen = associated.stream().min(Comparator.comparingDouble(index -> associations[index].distance()))
                        .orElseThrow();
            } else {
                chosen = associated.get(random.nextInt(associated.size()));
            }
            selected.add(pool.get(chosen));
            nicheCounts[direction]++;
            remaining.remove(Integer.valueOf(chosen));
        }
        return selected;
    }

    private static List<List<Integer>> nondominatedSort(List<Individual> population) {
        int size = population.size();
        List<List<Integer>> dominates = new ArrayList<>(size);
        int[] dominatedBy = new int[size];
        List<List<Integer>> fronts = new ArrayList<>();
        List<Integer> first = new ArrayList<>();
        for (int i = 0; i < size; i++) dominates.add(new ArrayList<>());
        for (int left = 0; left < size; left++) for (int right = left + 1; right < size; right++) {
            int comparison = compare(population.get(left).evaluation().objectives(),
                    population.get(right).evaluation().objectives());
            if (comparison < 0) {
                dominates.get(left).add(right);
                dominatedBy[right]++;
            } else if (comparison > 0) {
                dominates.get(right).add(left);
                dominatedBy[left]++;
            }
        }
        for (int index = 0; index < size; index++) if (dominatedBy[index] == 0) first.add(index);
        while (!first.isEmpty()) {
            fronts.add(first);
            List<Integer> next = new ArrayList<>();
            for (int index : first) for (int dominated : dominates.get(index)) {
                if (--dominatedBy[dominated] == 0) next.add(dominated);
            }
            first = next;
        }
        return fronts;
    }

    private static List<Integer> nondominatedFront(List<Individual> population) {
        return nondominatedSort(population).getFirst();
    }

    private static int compare(double[] left, double[] right) {
        boolean leftBetter = false, rightBetter = false;
        for (int i = 0; i < left.length; i++) {
            if (left[i] < right[i]) leftBetter = true;
            else if (left[i] > right[i]) rightBetter = true;
        }
        if (leftBetter == rightBetter) return 0;
        return leftBetter ? -1 : 1;
    }

    private static double[][] normalize(List<Individual> population) {
        int objectives = population.getFirst().evaluation().objectives().length;
        double[] minimum = new double[objectives], maximum = new double[objectives];
        Arrays.fill(minimum, Double.POSITIVE_INFINITY);
        Arrays.fill(maximum, Double.NEGATIVE_INFINITY);
        for (Individual individual : population) for (int objective = 0; objective < objectives; objective++) {
            double value = individual.evaluation().objectives()[objective];
            if (Double.isFinite(value)) {
                minimum[objective] = Math.min(minimum[objective], value);
                maximum[objective] = Math.max(maximum[objective], value);
            }
        }
        double[][] result = new double[population.size()][objectives];
        for (int row = 0; row < population.size(); row++) for (int objective = 0; objective < objectives; objective++) {
            double value = population.get(row).evaluation().objectives()[objective];
            double range = maximum[objective] - minimum[objective];
            result[row][objective] = Double.isFinite(value) && range > 0
                    ? (value - minimum[objective]) / range : 1.0;
        }
        return result;
    }

    private static Association[] associate(double[][] points, double[][] directions) {
        Association[] result = new Association[points.length];
        for (int point = 0; point < points.length; point++) {
            int nearest = 0;
            double nearestDistance = Double.POSITIVE_INFINITY;
            for (int direction = 0; direction < directions.length; direction++) {
                double dot = 0.0, norm = 0.0, pointNorm = 0.0;
                for (int objective = 0; objective < points[point].length; objective++) {
                    dot += points[point][objective] * directions[direction][objective];
                    norm += directions[direction][objective] * directions[direction][objective];
                    pointNorm += points[point][objective] * points[point][objective];
                }
                double distance = Math.sqrt(Math.max(0.0, pointNorm - dot * dot / norm));
                if (distance < nearestDistance) {
                    nearestDistance = distance;
                    nearest = direction;
                }
            }
            result[point] = new Association(nearest, nearestDistance);
        }
        return result;
    }

    private static int closestToIdeal(List<Individual> population) {
        double[][] normalized = normalize(population);
        int best = 0;
        double bestDistance = Double.POSITIVE_INFINITY;
        for (int i = 0; i < normalized.length; i++) {
            double distance = 0.0;
            for (double objective : normalized[i]) distance += objective * objective;
            if (distance < bestDistance) {
                bestDistance = distance;
                best = i;
            }
        }
        return best;
    }

    private static double[][] energyDirections(int count, int dimensions, long seed) {
        Random random = new Random(seed);
        double[][] points = new double[count][dimensions];
        for (int point = 0; point < count; point++) {
            double sum = 0.0;
            for (int dimension = 0; dimension < dimensions; dimension++) {
                points[point][dimension] = -Math.log(Math.max(random.nextDouble(), 1e-12));
                sum += points[point][dimension];
            }
            for (int dimension = 0; dimension < dimensions; dimension++) points[point][dimension] /= sum;
        }
        double exponent = dimensions * dimensions;
        double step = 0.02;
        for (int iteration = 0; iteration < 800; iteration++) {
            double[][] gradient = new double[count][dimensions];
            for (int left = 0; left < count; left++) for (int right = left + 1; right < count; right++) {
                double distanceSquared = 0.0;
                for (int dimension = 0; dimension < dimensions; dimension++) {
                    double delta = points[left][dimension] - points[right][dimension];
                    distanceSquared += delta * delta;
                }
                double scale = Math.min(1e4, exponent
                        * Math.pow(Math.max(distanceSquared, 1e-2), -0.5 * exponent - 1.0));
                for (int dimension = 0; dimension < dimensions; dimension++) {
                    double force = scale * (points[left][dimension] - points[right][dimension]);
                    gradient[left][dimension] -= force;
                    gradient[right][dimension] += force;
                }
            }
            for (int point = 0; point < count; point++) {
                double largest = Arrays.stream(gradient[point]).map(Math::abs).max().orElse(1.0);
                double mean = Arrays.stream(gradient[point]).average().orElse(0.0);
                double total = 0.0;
                for (int dimension = 0; dimension < dimensions; dimension++) {
                    points[point][dimension] = Math.max(1e-5, points[point][dimension]
                            - step * (gradient[point][dimension] - mean) / Math.max(1.0, largest));
                    total += points[point][dimension];
                }
                for (int dimension = 0; dimension < dimensions; dimension++) points[point][dimension] /= total;
            }
            step *= 0.997;
        }
        return points;
    }

    private static int[] selected(boolean[] genome, int featureCount) {
        int count = 0;
        for (int i = 0; i < featureCount; i++) if (genome[i]) count++;
        int[] result = new int[count];
        for (int i = 0, index = 0; i < featureCount; i++) if (genome[i]) result[index++] = i;
        return result;
    }

    private static String key(boolean[] genome) {
        StringBuilder result = new StringBuilder(genome.length);
        for (boolean bit : genome) result.append(bit ? '1' : '0');
        return result.toString();
    }
}
