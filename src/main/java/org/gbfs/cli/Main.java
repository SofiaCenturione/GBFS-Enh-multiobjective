package org.gbfs.cli;

import org.gbfs.data.Dataset;
import org.gbfs.data.MatDatasetLoader;
import org.gbfs.featureselection.EnhancedFeatureSelector;
import org.gbfs.featureselection.WekaKnnClassifier;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

public final class Main {
    private static final String[] FRONT_COLUMNS = {
            "dataset_idx", "dataset", "method", "run", "point_id", "fnum", "fRatio",
            "acc_train_cv", "eRate_train_cv", "red", "fset"
    };
    private static final String[] TEST_COLUMNS = {
            "dataset_idx", "dataset", "method", "run", "test_mode", "acc_test", "eRate_test",
            "fnum", "fRatio", "red", "time_total", "fset"
    };

    private Main() {}

    public static void main(String[] args) {
        int status = run(args);
        if (status != 0) {
            System.exit(status);
        }
    }

    public static int run(String[] args) {
        Map<String, String> options;
        try {
            options = parse(args);
        } catch (IllegalArgumentException ex) {
            System.err.println(ex.getMessage());
            return 2;
        }

        int datasetIndex = integer(options, "dataset_idx", -1);
        int runId = integer(options, "run", 1);
        String algo = options.getOrDefault("algo", "");
        Path outRoot = Path.of(options.getOrDefault("out_root", "results"));
        long seed = 1000L * datasetIndex + runId;
        Path runDir = outRoot.resolve(String.format(Locale.ROOT, "dataset_%02d", datasetIndex))
                .resolve(algoTag(algo, options))
                .resolve(String.format(Locale.ROOT, "run_%02d", runId));

        try {
            Files.createDirectories(runDir);
            clearPreviousRunArtifacts(runDir);
            Map<String, Object> runInfo = runInfo(options, datasetIndex, runId, algo, seed);
            writeJson(runDir.resolve("run_info.json"), runInfo);

            if (!supported(algo)) {
                writeJson(runDir.resolve("run_meta.json"), Map.of(
                        "algo", algo, "dataset_idx", datasetIndex, "run", runId,
                        "status", "unsupported"));
                Files.writeString(runDir.resolve("error.txt"), unsupportedMessage(algo));
                System.err.println("Unsupported algorithm; details written to " + runDir.resolve("error.txt"));
                return 2;
            }

            Path dataDir = Path.of(options.getOrDefault("data_dir", "Data"));
            Dataset dataset = new MatDatasetLoader(dataDir).load(datasetIndex);
            execute(options, runDir, dataset, datasetIndex, runId, seed, algo);
            System.out.println("DONE: " + runDir.toAbsolutePath());
            return 0;
        } catch (Exception ex) {
            try {
                writeJson(runDir.resolve("run_meta.json"), Map.of(
                        "algo", algo, "dataset_idx", datasetIndex, "run", runId, "status", "failed"));
                Files.writeString(runDir.resolve("error.txt"),
                        ex.getClass().getSimpleName() + ": " + String.valueOf(ex.getMessage()) + System.lineSeparator());
            } catch (IOException writeFailure) {
                System.err.println("Could not write failure artifacts: " + writeFailure.getMessage());
            }
            System.err.println("Run failed: " + ex.getMessage());
            return 1;
        }
    }

    private static void clearPreviousRunArtifacts(Path runDir) throws IOException {
        for (String file : List.of(
                "run_info.json",
                "run_meta.json",
                "front_train_cv.csv",
                "test_points.csv",
                "selected_features.txt",
                "selected_features_post_normal.txt",
                "selected_features_post_buddy.txt",
                "error.txt")) {
            Files.deleteIfExists(runDir.resolve(file));
        }
        for (String postMode : List.of("post_normal", "post_buddy")) {
            Path selection = runDir.resolve(postMode).resolve("selected_features.txt");
            Files.deleteIfExists(selection);
            Path postDirectory = selection.getParent();
            if (Files.isDirectory(postDirectory)) {
                try (var entries = Files.list(postDirectory)) {
                    if (entries.findAny().isEmpty()) Files.delete(postDirectory);
                }
            }
        }
    }

    private static boolean supported(String algo) {
        if (algo.equals("all") || algo.startsWith("enh:")) {
            return true;
        }
        if (!algo.startsWith("trad:")) return false;
        return List.of("FILTER_pearson", "FILTER_kendall", "FILTER_reliefF",
                        "FILTER_chi2", "FILTER_info_gain")
                .contains(algo.substring("trad:".length()));
    }

    private static String unsupportedMessage(String algo) {
        return "This Java port does not yet implement '" + algo + "'. No success metrics were generated. "
                + "Supported Java algorithms: all, enh:<knn|local_thresh|fisher_red_deg|mst_plus_local|probabilistic>:<normal|rc_greedy>, "
                + "trad:FILTER_pearson, trad:FILTER_kendall, trad:FILTER_reliefF, trad:FILTER_chi2, trad:FILTER_info_gain.";
    }

    private static void execute(Map<String, String> options, Path runDir, Dataset dataset,
                                int datasetIndex, int runId, long seed, String algo) throws Exception {
        double testSize = decimal(options, "test_size", 0.30);
        int splitSeed = integer(options, "split_seed", 42);
        int cvFolds = integer(options, "cv_folds", 5);
        int cvSeed = integer(options, "cv_seed", 42);
        int knnK = integer(options, "knn_eval_k", 5);
        if (!(testSize > 0 && testSize < 1) || cvFolds < 2 || knnK < 1) {
            throw new IllegalArgumentException("test_size must be in (0,1), cv_folds >= 2, and knn_eval_k >= 1.");
        }

        double[][] raw = dataset.values();
        double[] labels = dataset.labels();
        int[][] split = stratifiedSplit(labels, testSize, splitSeed);
        double[][] normalized = minMax(raw);
        if (algo.startsWith("enh:")) {
            executeEnhanced(options, runDir, dataset, datasetIndex, runId, seed, algo,
                    split, normalized, labels, testSize, splitSeed, cvFolds, cvSeed, knnK);
            return;
        }
        double[][] xTrain = standardize(selectRows(raw, split[0]), raw, split[0]);
        double[][] xTest = standardize(selectRows(raw, split[1]), raw, split[0]);
        int[] selected;
        String method;
        List<Integer> featureSetSizes = new ArrayList<>();
        double[] scores = null;
        long scoringStarted = System.nanoTime();

        if (algo.equals("all")) {
            method = "KNN_ALL";
            selected = range(dataset.featureCount());
            featureSetSizes.add(selected.length);
        } else {
            method = algo.substring("trad:".length());
            scores = traditionalScores(method, xTrain, labelsAt(labels, split[0]));
            int fixed = integer(options, "fixed_fnum", 0);
            List<Double> ratios = parseRatios(options.getOrDefault("ratios",
                    "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"));
            if (fixed > 0) {
                featureSetSizes.add(Math.min(fixed, dataset.featureCount()));
            } else {
                for (double ratio : ratios) {
                    if (!(ratio > 0 && ratio <= 1)) {
                        throw new IllegalArgumentException("Feature ratios must be in (0,1].");
                    }
                    featureSetSizes.add(Math.max(1, Math.min(dataset.featureCount(),
                            (int) Math.floor(ratio * dataset.featureCount() + 0.5))));
                }
            }
            int maxSize = featureSetSizes.stream().mapToInt(Integer::intValue).max().orElse(1);
            int[] ranking = rank(scores);
            selected = Arrays.copyOf(ranking, maxSize);
        }
        double scoreTime = (System.nanoTime() - scoringStarted) / 1e9;

        long start = System.nanoTime();
        List<Map<String, Object>> front = new ArrayList<>();
        Map<Integer, Map<String, Object>> bestBySize = new TreeMap<>();
        for (int size : featureSetSizes) {
            int[] subset = Arrays.copyOf(selected, size);
            double accuracy = crossValidation(xTrain, labelsAt(labels, split[0]), subset,
                    cvFolds, cvSeed, knnK);
            Map<String, Object> row = new LinkedHashMap<>();
            row.put("dataset_idx", datasetIndex);
            row.put("dataset", dataset.name());
            row.put("method", method);
            row.put("run", runId);
            row.put("point_id", featureSetSizes.indexOf(size));
            row.put("fnum", size);
            row.put("fRatio", (double) size / dataset.featureCount());
            row.put("acc_train_cv", accuracy);
            row.put("eRate_train_cv", 1.0 - accuracy);
            row.put("red", redundancy(normalized, subset));
            row.put("fset", featureString(subset));
            if (algo.equals("all")) {
                row.put("note", "ALL features");
                row.put("scaler", "StandardScaler");
            } else {
                row.put("ratio_grid", options.getOrDefault("ratios", "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"));
                row.put("score_time", scoreTime);
            }
            Map<String, Object> previous = bestBySize.get(size);
            if (previous == null || accuracy > (double) previous.get("acc_train_cv")) {
                bestBySize.put(size, row);
            }
        }
        front.addAll(bestBySize.values());

        List<Map<String, Object>> testPoints = new ArrayList<>();
        if (algo.equals("all")) {
            double accuracy = accuracy(xTrain, labelsAt(labels, split[0]), xTest,
                    labelsAt(labels, split[1]), selected, knnK);
            Map<String, Object> extras = new LinkedHashMap<>();
            extras.put("note", "ALL features");
            extras.put("scaler", "StandardScaler");
            testPoints.add(testRow(datasetIndex, dataset.name(), method, runId, "all",
                    selected, dataset.featureCount(), accuracy, redundancy(normalized, selected),
                    (System.nanoTime() - start) / 1e9, extras));
        } else {
            for (int size : featureSetSizes) {
                int[] subset = Arrays.copyOf(selected, size);
                double accuracy = accuracy(xTrain, labelsAt(labels, split[0]), xTest,
                        labelsAt(labels, split[1]), subset, knnK);
                boolean fixedSize = integer(options, "fixed_fnum", 0) > 0;
                Map<String, Object> extras = new LinkedHashMap<>();
                if (fixedSize) extras.put("fixed_fnum", size);
                else extras.put("ratio", (double) size / dataset.featureCount());
                extras.put("score_time", scoreTime);
                testPoints.add(testRow(datasetIndex, dataset.name(), method, runId, fixedSize ? "fixed_k" : "ratio",
                        subset, dataset.featureCount(), accuracy, redundancy(normalized, subset),
                        scoreTime + (System.nanoTime() - start) / 1e9, extras));
            }
        }

        writeCsv(runDir.resolve("front_train_cv.csv"), FRONT_COLUMNS, front);
        writeCsv(runDir.resolve("test_points.csv"), TEST_COLUMNS, testPoints);
        Map<String, Object> meta = new LinkedHashMap<>();
        meta.put("algo", algo);
        meta.put("method", method);
        meta.put("dataset_idx", datasetIndex);
        meta.put("dataset", dataset.name());
        meta.put("run", runId);
        meta.put("seed_run", seed);
        meta.put("test_size", testSize);
        meta.put("split_seed", splitSeed);
        meta.put("cv_folds", cvFolds);
        meta.put("cv_seed", cvSeed);
        meta.put("knn_eval_k", knnK);
        meta.put("status", "completed");
        writeJson(runDir.resolve("run_meta.json"), meta);
        if (algo.equals("all")) {
            Files.writeString(runDir.resolve("selected_features.txt"), featureString(selected));
        }
    }

    private static void executeEnhanced(Map<String, String> options, Path runDir, Dataset dataset,
                                        int datasetIndex, int runId, long seed, String algo,
                                        int[][] split, double[][] normalized, double[] labels,
                                        double testSize, int splitSeed, int cvFolds, int cvSeed,
                                        int knnK) throws Exception {
        String[] parts = algo.split(":", -1);
        if (parts.length != 3 || !parts[0].equals("enh")) {
            throw new IllegalArgumentException("Enhanced algorithm must be enh:<initializer>:<normal|rc_greedy>.");
        }
        String initializer = parts[1];
        String kshellMode = parts[2];
        double[][] train = selectRows(normalized, split[0]);
        double[][] test = selectRows(normalized, split[1]);
        double[] trainLabels = labelsAt(labels, split[0]);
        double[] testLabels = labelsAt(labels, split[1]);
        int population = integer(options, "base_pop", 20);
        int generations = integer(options, "base_gen", 50);
        int baseK = integer(options, "base_kneigh", 5);
        double beta = decimal(options, "prob_beta", 2.0);
        double rcTau = decimal(options, "ks_rc_tau", 0.3);
        double buddyLambda = decimal(options, "buddy_lambda", 0.5);
        int maxAdd = integer(options, "ks_max_add", 5);
        EnhancedFeatureSelector.Selection selection = EnhancedFeatureSelector.select(
                initializer, kshellMode, train, trainLabels, baseK, cvFolds, Math.toIntExact(seed),
                maxAdd, rcTau, buddyLambda, beta, population, generations);
        String method = "GBFS_enhanced__init_" + initializer + "__ks_" + kshellMode;

        Map<Integer, Map<String, Object>> bestByFeatureCount = new TreeMap<>();
        for (int point = 0; point < selection.pareto().length; point++) {
            int[] featureSet = selection.pareto()[point];
            double trainAccuracy = crossValidation(train, trainLabels, featureSet, cvFolds, cvSeed, knnK);
            Map<String, Object> frontRow = new LinkedHashMap<>();
            frontRow.put("dataset_idx", datasetIndex);
            frontRow.put("dataset", dataset.name());
            frontRow.put("method", method);
            frontRow.put("run", runId);
            frontRow.put("point_id", point);
            frontRow.put("fnum", featureSet.length);
            frontRow.put("fRatio", (double) featureSet.length / dataset.featureCount());
            frontRow.put("acc_train_cv", trainAccuracy);
            frontRow.put("eRate_train_cv", 1.0 - trainAccuracy);
            frontRow.put("red", redundancy(normalized, featureSet));
            frontRow.put("fset", featureString(featureSet));
            frontRow.put("init_mode", initializer);
            frontRow.put("ks_mode", kshellMode);
            frontRow.put("prob_beta", beta);
            frontRow.put("ks_rc_tau", rcTau);
            frontRow.put("buddy_lambda", buddyLambda);
            Map<String, Object> previous = bestByFeatureCount.get(featureSet.length);
            if (previous == null || trainAccuracy > (double) previous.get("acc_train_cv")) {
                bestByFeatureCount.put(featureSet.length, frontRow);
            }
        }
        List<Map<String, Object>> front = new ArrayList<>(bestByFeatureCount.values());

        List<Map<String, Object>> testPoints = new ArrayList<>();
        double normalTime = selection.solverSeconds();
        double buddyTime = normalTime + selection.buddySeconds();
        addEnhancedResult(testPoints, "normal", selection.postNormal(), train, trainLabels,
                test, testLabels, normalized, dataset, datasetIndex, runId, method, normalTime,
                initializer, kshellMode, beta, rcTau, buddyLambda, knnK);
        addEnhancedResult(testPoints, "buddy", selection.postBuddy(), train, trainLabels,
                test, testLabels, normalized, dataset, datasetIndex, runId, method, buddyTime,
                initializer, kshellMode, beta, rcTau, buddyLambda, knnK);

        writeCsv(runDir.resolve("front_train_cv.csv"), FRONT_COLUMNS, front);
        writeCsv(runDir.resolve("test_points.csv"), TEST_COLUMNS, testPoints);
        Map<String, Object> meta = new LinkedHashMap<>();
        meta.put("algo", algo);
        meta.put("method", method);
        meta.put("dataset_idx", datasetIndex);
        meta.put("dataset", dataset.name());
        meta.put("run", runId);
        meta.put("seed_run", seed);
        Map<String, Object> config = new LinkedHashMap<>();
        config.put("test_size", testSize);
        config.put("split_seed", splitSeed);
        config.put("cv_folds", cvFolds);
        config.put("cv_seed", cvSeed);
        config.put("knn_eval_k", knnK);
        config.put("pop", population);
        config.put("gen", generations);
        config.put("k_neigh", baseK);
        config.put("ks_max_add", maxAdd);
        config.put("ks_rc_tau", rcTau);
        config.put("prob_beta", beta);
        config.put("buddy_lambda", buddyLambda);
        meta.put("cfg", config);
        Path comboDir = enhancedComboDirectory(runDir, datasetIndex, initializer, kshellMode);
        meta.put("paths", Map.of("combo_dir", comboDir.toString()));
        writeJson(runDir.resolve("run_meta.json"), meta);
        writeEnhancedRawArtifacts(comboDir, runDir, datasetIndex, runId, initializer, kshellMode,
                dataset.featureCount(), selection, testPoints, normalTime, buddyTime);
    }

    private static void addEnhancedResult(List<Map<String, Object>> rows, String postMode,
                                         int[] selected, double[][] train, double[] trainLabels,
                                         double[][] test, double[] testLabels, double[][] normalized,
                                         Dataset dataset, int datasetIndex, int runId, String method,
                                         double elapsed, String initializer, String kshellMode,
                                         double beta, double rcTau, double buddyLambda, int knnK) throws Exception {
        double accuracy = accuracy(train, trainLabels, test, testLabels, selected, knnK);
        Map<String, Object> extra = new LinkedHashMap<>();
        extra.put("init_mode", initializer);
        extra.put("ks_mode", kshellMode);
        extra.put("post_mode", postMode);
        extra.put("prob_beta", beta);
        extra.put("ks_rc_tau", rcTau);
        extra.put("buddy_lambda", buddyLambda);
        rows.add(testRow(datasetIndex, dataset.name(), method, runId, "selected", selected,
                dataset.featureCount(), accuracy, redundancy(normalized, selected), elapsed, extra));
    }

    private static Path enhancedComboDirectory(Path runDir, int datasetIndex,
                                              String initializer, String kshellMode) {
        return runDir.resolve("gbfs_enhanced_raw")
                .resolve(String.format(Locale.ROOT, "%02d", datasetIndex))
                .resolve("run_01")
                .resolve(initializer)
                .resolve("ks_" + kshellMode);
    }

    private static void writeEnhancedRawArtifacts(
            Path comboDir, Path runDir, int datasetIndex, int runId, String initializer,
            String kshellMode, int featureCount, EnhancedFeatureSelector.Selection selection,
            List<Map<String, Object>> testPoints, double normalTime, double buddyTime) throws IOException {
        Files.createDirectories(comboDir);
        writeNpyBooleanMatrix(comboDir.resolve("pareto_masks.npy"),
                selection.pareto(), featureCount);
        writeNpyDoubleMatrix(comboDir.resolve("pareto_objs.npy"), selection.paretoObjectives());
        writeNpyLongVector(comboDir.resolve("kNeigh_chosen.npy"), selection.neighborMask());
        Files.writeString(comboDir.resolve("S0_selected.txt"), featureString(selection.core()));
        writeJson(comboDir.resolve("solver_meta.json"), Map.of(
                "run", 1,
                "init_mode", initializer,
                "kshell_seq_mode", kshellMode,
                "log_id", 1,
                "assi_num", 0.0,
                "solver_time", normalTime,
                "fnum_S0", selection.core().length));

        writePostArtifacts(comboDir.resolve("post_normal"), testPoints.get(0),
                initializer, kshellMode, normalTime, 0.0, selection.postNormal());
        writePostArtifacts(comboDir.resolve("post_buddy"), testPoints.get(1),
                initializer, kshellMode, normalTime, buddyTime - normalTime, selection.postBuddy());
        writeAblationSummary(runDir.resolve("gbfs_enhanced_raw")
                        .resolve(String.format(Locale.ROOT, "%02d", datasetIndex))
                        .resolve("ablation_summary.csv"),
                runId, initializer, kshellMode, testPoints);
    }

    private static void writePostArtifacts(Path postDir, Map<String, Object> testPoint,
                                           String initializer, String kshellMode,
                                           double solverTime, double postExtraTime,
                                           int[] selected) throws IOException {
        Files.createDirectories(postDir);
        Files.writeString(postDir.resolve("selected_features.txt"), featureString(selected));
        Map<String, Object> metrics = new LinkedHashMap<>();
        metrics.put("run", 1);
        metrics.put("init_mode", initializer);
        metrics.put("kshell_seq_mode", kshellMode);
        metrics.put("post_mode", testPoint.get("post_mode"));
        metrics.put("acc_test", testPoint.get("acc_test"));
        metrics.put("red", testPoint.get("red"));
        metrics.put("fnum", testPoint.get("fnum"));
        metrics.put("time_total", testPoint.get("time_total"));
        metrics.put("time_solver", solverTime);
        metrics.put("time_post_extra", postExtraTime);
        writeJson(postDir.resolve("post_metrics.json"), metrics);
    }

    private static void writeAblationSummary(Path path, int runId, String initializer,
                                             String kshellMode,
                                             List<Map<String, Object>> testPoints) throws IOException {
        List<String> columns = new ArrayList<>();
        columns.add("run");
        for (String postMode : List.of("normal", "buddy")) {
            String prefix = initializer + "__ks" + kshellMode + "__post" + postMode;
            columns.add("acc_" + prefix);
            columns.add("fnum_" + prefix);
            columns.add("red_" + prefix);
            columns.add("time_" + prefix);
        }
        List<Map<String, Object>> rows = new ArrayList<>();
        Map<String, Object> run = ablationRow(Integer.toString(runId), testPoints);
        rows.add(run);
        rows.add(ablationRow("mean", testPoints));
        Map<String, Object> std = new LinkedHashMap<>();
        std.put("run", "std");
        for (int i = 1; i < columns.size(); i++) std.put(columns.get(i), 0.0);
        rows.add(std);
        writeCsv(path, columns.toArray(String[]::new), rows);
    }

    private static Map<String, Object> ablationRow(String label, List<Map<String, Object>> testPoints) {
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("run", label);
        for (Map<String, Object> point : testPoints) {
            String postMode = point.get("post_mode").toString();
            String prefix = point.get("init_mode") + "__ks" + point.get("ks_mode") + "__post" + postMode;
            row.put("acc_" + prefix, point.get("acc_test"));
            row.put("fnum_" + prefix, point.get("fnum"));
            row.put("red_" + prefix, point.get("red"));
            row.put("time_" + prefix, point.get("time_total"));
        }
        return row;
    }

    private static void writeNpyBooleanMatrix(Path path, int[][] featureSets, int featureCount)
            throws IOException {
        ByteBuffer payload = ByteBuffer.allocate(featureSets.length * featureCount);
        for (int[] featureSet : featureSets) {
            boolean[] selected = new boolean[featureCount];
            for (int feature : featureSet) if (feature >= 0 && feature < featureCount) selected[feature] = true;
            for (boolean value : selected) payload.put((byte) (value ? 1 : 0));
        }
        writeNpy(path, "|b1", "(" + featureSets.length + ", " + featureCount + ")",
                payload.array());
    }

    private static void writeNpyDoubleMatrix(Path path, double[][] values) throws IOException {
        int columns = values.length == 0 ? 0 : values[0].length;
        ByteBuffer payload = ByteBuffer.allocate(values.length * columns * Double.BYTES)
                .order(ByteOrder.LITTLE_ENDIAN);
        for (double[] row : values) {
            if (row.length != columns) throw new IllegalArgumentException("Expected rectangular objective matrix.");
            for (double value : row) payload.putDouble(value);
        }
        writeNpy(path, "<f8", "(" + values.length + ", " + columns + ")", payload.array());
    }

    private static void writeNpyLongVector(Path path, boolean[] values) throws IOException {
        ByteBuffer payload = ByteBuffer.allocate(values.length * Long.BYTES)
                .order(ByteOrder.LITTLE_ENDIAN);
        for (boolean value : values) payload.putLong(value ? 1L : 0L);
        writeNpy(path, "<i8", "(" + values.length + ",)", payload.array());
    }

    private static void writeNpy(Path path, String descriptor, String shape, byte[] payload)
            throws IOException {
        String header = "{'descr': '" + descriptor + "', 'fortran_order': False, 'shape': "
                + shape + ", }";
        int padding = 16 - ((10 + header.length() + 1) % 16);
        if (padding == 16) padding = 0;
        header += " ".repeat(padding) + "\n";
        byte[] headerBytes = header.getBytes(StandardCharsets.US_ASCII);
        ByteBuffer output = ByteBuffer.allocate(10 + headerBytes.length + payload.length)
                .order(ByteOrder.LITTLE_ENDIAN);
        output.put((byte) 0x93).put("NUMPY".getBytes(StandardCharsets.US_ASCII));
        output.put((byte) 1).put((byte) 0).putShort((short) headerBytes.length);
        output.put(headerBytes).put(payload);
        Files.write(path, output.array());
    }

    private static Map<String, Object> testRow(int datasetIndex, String dataset, String method, int runId,
                                                String mode, int[] subset, int featureCount, double accuracy,
                                                double redundancy, double seconds, Map<String, Object> extra) {
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("dataset_idx", datasetIndex);
        row.put("dataset", dataset);
        row.put("method", method);
        row.put("run", runId);
        row.put("test_mode", mode);
        row.put("acc_test", accuracy);
        row.put("eRate_test", 1.0 - accuracy);
        row.put("fnum", subset.length);
        row.put("fRatio", (double) subset.length / featureCount);
        row.put("red", redundancy);
        row.put("time_total", seconds);
        row.put("fset", featureString(subset));
        row.putAll(extra);
        return row;
    }

    private static double crossValidation(double[][] x, double[] y, int[] subset,
                                          int folds, int seed, int k) throws Exception {
        int[][] foldIndices = stratifiedFolds(y, folds, seed);
        double total = 0;
        for (int[] validation : foldIndices) {
            boolean[] isValidation = new boolean[y.length];
            for (int index : validation) isValidation[index] = true;
            int[] training = new int[y.length - validation.length];
            for (int i = 0, pos = 0; i < y.length; i++) if (!isValidation[i]) training[pos++] = i;
            double[][] trainX = selectRows(x, training);
            double[][] validX = selectRows(x, validation);
            double[] trainY = labelsAt(y, training);
            double[] validY = labelsAt(y, validation);
            total += accuracy(trainX, trainY, validX, validY, subset, k);
        }
        return foldIndices.length == 0 ? 0 : total / foldIndices.length;
    }

    private static double accuracy(double[][] train, double[] trainY, double[][] test, double[] testY,
                                   int[] subset, int k) throws Exception {
        return WekaKnnClassifier.accuracy(train, trainY, test, testY, subset, k);
    }

    private static int[][] stratifiedSplit(double[] labels, double testSize, int seed) {
        Map<Double, List<Integer>> classes = grouped(labels);
        int testCount = Math.max(1, (int) Math.ceil(labels.length * testSize));
        if (labels.length - testCount < classes.size()) {
            throw new IllegalArgumentException("Not enough samples for a stratified train/test split.");
        }
        Random random = new Random(seed);
        Map<Double, Integer> testCounts = allocate(testCount, classes);
        List<Integer> train = new ArrayList<>(), test = new ArrayList<>();
        for (Map.Entry<Double, List<Integer>> entry : classes.entrySet()) {
            List<Integer> indices = entry.getValue();
            java.util.Collections.shuffle(indices, random);
            int count = testCounts.get(entry.getKey());
            test.addAll(indices.subList(0, count));
            train.addAll(indices.subList(count, indices.size()));
        }
        java.util.Collections.shuffle(train, random);
        java.util.Collections.shuffle(test, random);
        return new int[][] { train.stream().mapToInt(Integer::intValue).toArray(),
                test.stream().mapToInt(Integer::intValue).toArray() };
    }

    private static int[][] stratifiedFolds(double[] labels, int folds, int seed) {
        Map<Double, List<Integer>> classes = grouped(labels);
        for (List<Integer> indices : classes.values()) {
            if (indices.size() < folds) throw new IllegalArgumentException("Each class must have at least cv_folds samples.");
        }
        List<List<Integer>> result = new ArrayList<>();
        for (int i = 0; i < folds; i++) result.add(new ArrayList<>());
        Random random = new Random(seed);
        int fold = 0;
        for (List<Integer> indices : classes.values()) {
            java.util.Collections.shuffle(indices, random);
            for (int index : indices) {
                result.get(fold++ % folds).add(index);
            }
        }
        return result.stream().map(list -> list.stream().mapToInt(Integer::intValue).toArray()).toArray(int[][]::new);
    }

    private static Map<Double, Integer> allocate(int total, Map<Double, List<Integer>> classes) {
        Map<Double, Integer> counts = new TreeMap<>();
        int assigned = 0;
        for (Map.Entry<Double, List<Integer>> entry : classes.entrySet()) {
            int count = (int) Math.floor((double) total * entry.getValue().size()
                    / classes.values().stream().mapToInt(List::size).sum());
            count = Math.min(count, entry.getValue().size() - 1);
            counts.put(entry.getKey(), count);
            assigned += count;
        }
        while (assigned < total) {
            Double best = null;
            double bestRemainder = Double.NEGATIVE_INFINITY;
            for (Map.Entry<Double, List<Integer>> entry : classes.entrySet()) {
                int current = counts.get(entry.getKey());
                if (current >= entry.getValue().size() - 1) continue;
                double ideal = (double) total * entry.getValue().size()
                        / classes.values().stream().mapToInt(List::size).sum();
                double remainder = ideal - current;
                if (remainder > bestRemainder) {
                    best = entry.getKey();
                    bestRemainder = remainder;
                }
            }
            if (best == null) throw new IllegalArgumentException("Cannot allocate a stratified test split.");
            counts.put(best, counts.get(best) + 1);
            assigned++;
        }
        return counts;
    }

    private static Map<Double, List<Integer>> grouped(double[] labels) {
        Map<Double, List<Integer>> groups = new TreeMap<>();
        for (int i = 0; i < labels.length; i++) groups.computeIfAbsent(labels[i], ignored -> new ArrayList<>()).add(i);
        return groups;
    }

    private static double[] pearsonScores(double[][] x, double[] labels, int[] indices) {
        double[] trainLabels = labelsAt(labels, indices);
        double[] scores = new double[x[0].length];
        for (int feature = 0; feature < scores.length; feature++) {
            double[] column = new double[x.length];
            for (int row = 0; row < x.length; row++) column[row] = x[row][feature];
            scores[feature] = Math.abs(correlation(column, trainLabels));
        }
        return scores;
    }

    private static double[] kendallScores(double[][] x, double[] labels, int[] indices) {
        double[] trainLabels = labelsAt(labels, indices);
        double[] scores = new double[x[0].length];
        for (int feature = 0; feature < scores.length; feature++) {
            long concordant = 0, discordant = 0, tiedX = 0, tiedY = 0;
            for (int left = 0; left < x.length; left++) for (int right = left + 1; right < x.length; right++) {
                int sx = Double.compare(x[left][feature], x[right][feature]);
                int sy = Double.compare(trainLabels[left], trainLabels[right]);
                if (sx == 0 && sy == 0) continue;
                if (sx == 0) tiedX++;
                else if (sy == 0) tiedY++;
                else if (sx == sy) concordant++;
                else discordant++;
            }
            double denominator = Math.sqrt((double) (concordant + discordant + tiedX)
                    * (concordant + discordant + tiedY));
            scores[feature] = denominator == 0 ? 0 : Math.abs((concordant - discordant) / denominator);
        }
        return scores;
    }

    private static double[] traditionalScores(String method, double[][] x, double[] labels) {
        int[] indices = range(labels.length);
        return switch (method) {
            case "FILTER_pearson" -> pearsonScores(x, labels, indices);
            case "FILTER_kendall" -> kendallScores(x, labels, indices);
            case "FILTER_reliefF" -> reliefScores(x, labels);
            case "FILTER_chi2" -> chiSquareScores(x, labels);
            case "FILTER_info_gain" -> informationGainScores(x, labels);
            default -> throw new IllegalArgumentException("Unsupported traditional method: " + method);
        };
    }

    private static double[] reliefScores(double[][] x, double[] labels) {
        int rows = x.length;
        int features = x[0].length;
        double[] scores = new double[features];
        Map<Double, Integer> classCounts = new TreeMap<>();
        for (double label : labels) classCounts.merge(label, 1, Integer::sum);
        for (int row = 0; row < rows; row++) {
            int hit = -1;
            double hitDistance = Double.POSITIVE_INFINITY;
            Map<Double, Integer> misses = new TreeMap<>();
            Map<Double, Double> missDistances = new TreeMap<>();
            for (int candidate = 0; candidate < rows; candidate++) {
                if (candidate == row) continue;
                double distance = 0;
                for (int feature = 0; feature < features; feature++) {
                    double delta = x[row][feature] - x[candidate][feature];
                    distance += delta * delta;
                }
                if (labels[candidate] == labels[row]) {
                    if (distance < hitDistance) {
                        hitDistance = distance;
                        hit = candidate;
                    }
                } else if (distance < missDistances.getOrDefault(labels[candidate], Double.POSITIVE_INFINITY)) {
                    misses.put(labels[candidate], candidate);
                    missDistances.put(labels[candidate], distance);
                }
            }
            if (hit < 0 || misses.isEmpty()) continue;
            for (int feature = 0; feature < features; feature++) {
                double hitDifference = Math.abs(x[row][feature] - x[hit][feature]);
                double missDifference = 0.0;
                for (Map.Entry<Double, Integer> miss : misses.entrySet()) {
                    double prior = (double) classCounts.get(miss.getKey()) / rows;
                    missDifference += prior / Math.max(1.0 - (double) classCounts.get(labels[row]) / rows, 1e-12)
                            * Math.abs(x[row][feature] - x[miss.getValue()][feature]);
                }
                scores[feature] += (missDifference - hitDifference) / rows;
            }
        }
        return scores;
    }

    private static double[] chiSquareScores(double[][] x, double[] labels) {
        int bins = Math.max(2, Math.min(10, (int) Math.sqrt(x.length)));
        double[] scores = new double[x[0].length];
        Map<Double, Integer> classIndex = new TreeMap<>();
        for (double label : labels) classIndex.computeIfAbsent(label, ignored -> classIndex.size());
        int classes = classIndex.size();
        int[] classTotals = new int[classes];
        for (double label : labels) classTotals[classIndex.get(label)]++;
        for (int feature = 0; feature < scores.length; feature++) {
            double minimum = Double.POSITIVE_INFINITY, maximum = Double.NEGATIVE_INFINITY;
            for (double[] row : x) {
                minimum = Math.min(minimum, row[feature]);
                maximum = Math.max(maximum, row[feature]);
            }
            int[][] observed = new int[bins][classes];
            int[] binTotals = new int[bins];
            for (int row = 0; row < x.length; row++) {
                int bin = maximum == minimum ? 0
                        : Math.min(bins - 1, (int) ((x[row][feature] - minimum) / (maximum - minimum) * bins));
                observed[bin][classIndex.get(labels[row])]++;
                binTotals[bin]++;
            }
            for (int bin = 0; bin < bins; bin++) for (int cls = 0; cls < classes; cls++) {
                double expected = (double) binTotals[bin] * classTotals[cls] / x.length;
                if (expected > 0) {
                    double difference = observed[bin][cls] - expected;
                    scores[feature] += difference * difference / expected;
                }
            }
        }
        return scores;
    }

    private static double[] informationGainScores(double[][] x, double[] labels) {
        int bins = Math.max(2, Math.min(10, (int) Math.sqrt(x.length)));
        Map<Double, Integer> classIndex = new TreeMap<>();
        for (double label : labels) classIndex.computeIfAbsent(label, ignored -> classIndex.size());
        int classes = classIndex.size();
        int[] classTotals = new int[classes];
        for (double label : labels) classTotals[classIndex.get(label)]++;
        double labelEntropy = entropy(classTotals, labels.length);
        double[] scores = new double[x[0].length];
        for (int feature = 0; feature < scores.length; feature++) {
            double minimum = Double.POSITIVE_INFINITY, maximum = Double.NEGATIVE_INFINITY;
            for (double[] row : x) {
                minimum = Math.min(minimum, row[feature]);
                maximum = Math.max(maximum, row[feature]);
            }
            int[][] counts = new int[bins][classes];
            int[] binTotals = new int[bins];
            for (int row = 0; row < x.length; row++) {
                int bin = maximum == minimum ? 0
                        : Math.min(bins - 1, (int) ((x[row][feature] - minimum) / (maximum - minimum) * bins));
                counts[bin][classIndex.get(labels[row])]++;
                binTotals[bin]++;
            }
            double conditionalEntropy = 0.0;
            for (int bin = 0; bin < bins; bin++) {
                if (binTotals[bin] > 0) {
                    conditionalEntropy += (double) binTotals[bin] / labels.length
                            * entropy(counts[bin], binTotals[bin]);
                }
            }
            scores[feature] = labelEntropy - conditionalEntropy;
        }
        return scores;
    }

    private static double entropy(int[] counts, int total) {
        double result = 0.0;
        for (int count : counts) {
            if (count == 0) continue;
            double probability = (double) count / total;
            result -= probability * Math.log(probability);
        }
        return result;
    }

    private static double correlation(double[] left, double[] right) {
        double leftMean = Arrays.stream(left).average().orElse(0);
        double rightMean = Arrays.stream(right).average().orElse(0);
        double covariance = 0, leftVar = 0, rightVar = 0;
        for (int i = 0; i < left.length; i++) {
            double a = left[i] - leftMean, b = right[i] - rightMean;
            covariance += a * b;
            leftVar += a * a;
            rightVar += b * b;
        }
        double denominator = Math.sqrt(leftVar * rightVar);
        return denominator == 0 ? 0 : covariance / denominator;
    }

    private static int[] rank(double[] scores) {
        Integer[] order = new Integer[scores.length];
        for (int i = 0; i < order.length; i++) order[i] = i;
        Arrays.sort(order, Comparator.<Integer>comparingDouble(index -> -Math.abs(scores[index]))
                .thenComparingInt(Integer::intValue));
        return Arrays.stream(order).mapToInt(Integer::intValue).toArray();
    }

    private static double redundancy(double[][] normalized, int[] subset) {
        if (subset.length <= 1) return 0;
        double sum = 0;
        int pairs = 0;
        for (int a = 0; a < subset.length; a++) for (int b = a + 1; b < subset.length; b++) {
            double[] left = new double[normalized.length], right = new double[normalized.length];
            for (int row = 0; row < normalized.length; row++) {
                left[row] = normalized[row][subset[a]];
                right[row] = normalized[row][subset[b]];
            }
            sum += Math.abs(correlation(left, right));
            pairs++;
        }
        return sum / pairs;
    }

    private static double[][] standardize(double[][] rows, double[][] full, int[] trainIndices) {
        int columns = full[0].length;
        double[] means = new double[columns], deviations = new double[columns];
        for (int index : trainIndices) for (int column = 0; column < columns; column++) means[column] += full[index][column];
        for (int column = 0; column < columns; column++) means[column] /= trainIndices.length;
        for (int index : trainIndices) for (int column = 0; column < columns; column++) {
            double delta = full[index][column] - means[column];
            deviations[column] += delta * delta;
        }
        for (int column = 0; column < columns; column++) deviations[column] = Math.sqrt(deviations[column] / trainIndices.length);
        double[][] scaled = new double[rows.length][columns];
        for (int row = 0; row < rows.length; row++) for (int column = 0; column < columns; column++) {
            scaled[row][column] = deviations[column] == 0 ? 0 : (rows[row][column] - means[column]) / deviations[column];
        }
        return scaled;
    }

    private static double[][] minMax(double[][] values) {
        int columns = values[0].length;
        double[] min = new double[columns], max = new double[columns];
        Arrays.fill(min, Double.POSITIVE_INFINITY);
        Arrays.fill(max, Double.NEGATIVE_INFINITY);
        for (double[] row : values) for (int c = 0; c < columns; c++) {
            min[c] = Math.min(min[c], row[c]);
            max[c] = Math.max(max[c], row[c]);
        }
        double[][] scaled = new double[values.length][columns];
        for (int r = 0; r < values.length; r++) for (int c = 0; c < columns; c++) {
            scaled[r][c] = max[c] == min[c] ? 0 : (values[r][c] - min[c]) / (max[c] - min[c]);
        }
        return scaled;
    }

    private static double[][] selectRows(double[][] source, int[] indices) {
        double[][] result = new double[indices.length][];
        for (int i = 0; i < indices.length; i++) result[i] = source[indices[i]].clone();
        return result;
    }

    private static double[] labelsAt(double[] labels, int[] indices) {
        double[] result = new double[indices.length];
        for (int i = 0; i < indices.length; i++) result[i] = labels[indices[i]];
        return result;
    }

    private static int[] range(int size) {
        int[] result = new int[size];
        for (int i = 0; i < size; i++) result[i] = i;
        return result;
    }

    private static String featureString(int[] subset) {
        return Arrays.stream(subset).mapToObj(Integer::toString).collect(java.util.stream.Collectors.joining(" "));
    }

    private static List<Double> parseRatios(String value) {
        List<Double> result = new ArrayList<>();
        for (String part : value.split(",")) if (!part.isBlank()) result.add(Double.parseDouble(part.trim()));
        if (result.isEmpty()) throw new IllegalArgumentException("At least one ratio is required.");
        return result;
    }

    private static Map<String, Object> runInfo(Map<String, String> options, int datasetIndex, int runId,
                                               String algo, long seed) {
        Map<String, Object> info = new LinkedHashMap<>();
        info.put("dataset_idx", datasetIndex);
        info.put("run", runId);
        info.put("algo", algo);
        info.put("seed_run", seed);
        info.put("test_size", decimal(options, "test_size", 0.30));
        info.put("split_seed", integer(options, "split_seed", 42));
        info.put("cv_folds", integer(options, "cv_folds", 5));
        info.put("cv_seed", integer(options, "cv_seed", 42));
        info.put("knn_eval_k", integer(options, "knn_eval_k", 5));
        info.put("fixed_fnum", integer(options, "fixed_fnum", 0));
        info.put("prob_beta", decimal(options, "prob_beta", 2));
        info.put("ks_rc_tau", decimal(options, "ks_rc_tau", 0.3));
        info.put("buddy_lambda", decimal(options, "buddy_lambda", 0.5));
        return info;
    }

    private static String algoTag(String algo, Map<String, String> options) {
        String tag = algo.replace(":", "__").replace("/", "_").replace("\\", "_").replace(" ", "_");
        if (algo.startsWith("enh:")) {
            tag += "__beta_" + floatTag(decimal(options, "prob_beta", 2))
                    + "__tau_" + floatTag(decimal(options, "ks_rc_tau", 0.3))
                    + "__lam_" + floatTag(decimal(options, "buddy_lambda", 0.5));
        }
        return tag;
    }

    private static String floatTag(double value) {
        String result = String.format(Locale.ROOT, "%s", value == (long) value
                ? Long.toString((long) value) : Double.toString(value));
        return result.replace("-", "m").replace(".", "p");
    }

    private static void writeCsv(Path path, String[] columns, List<Map<String, Object>> rows) throws IOException {
        List<String> allColumns = new ArrayList<>(Arrays.asList(columns));
        for (Map<String, Object> row : rows) for (String key : row.keySet()) {
            if (!allColumns.contains(key)) allColumns.add(key);
        }
        StringBuilder csv = new StringBuilder(String.join(",", allColumns)).append("\n");
        for (Map<String, Object> row : rows) {
            for (int column = 0; column < allColumns.size(); column++) {
                if (column > 0) csv.append(',');
                Object value = row.get(allColumns.get(column));
                csv.append(csvEscape(value == null ? "" : value.toString()));
            }
            csv.append('\n');
        }
        Files.writeString(path, csv);
    }

    private static String csvEscape(String value) {
        if (value.contains(",") || value.contains("\"") || value.contains("\n")) {
            return "\"" + value.replace("\"", "\"\"") + "\"";
        }
        return value;
    }

    private static void writeJson(Path path, Map<String, ?> values) throws IOException {
        StringBuilder json = new StringBuilder("{\n");
        int index = 0;
        for (Map.Entry<String, ?> entry : values.entrySet()) {
            if (index++ > 0) json.append(",\n");
            json.append("  \"").append(jsonEscape(entry.getKey())).append("\": ")
                    .append(jsonValue(entry.getValue()));
        }
        json.append("\n}\n");
        Files.writeString(path, json);
    }

    private static String jsonValue(Object value) {
        if (value == null) return "null";
        if (value instanceof Number || value instanceof Boolean) return value.toString();
        if (value instanceof Map<?, ?> map) {
            List<String> entries = new ArrayList<>();
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                entries.add("\"" + jsonEscape(String.valueOf(entry.getKey())) + "\": "
                        + jsonValue(entry.getValue()));
            }
            return "{ " + String.join(", ", entries) + " }";
        }
        if (value instanceof Iterable<?> values) {
            List<String> items = new ArrayList<>();
            for (Object item : values) items.add(jsonValue(item));
            return "[" + String.join(", ", items) + "]";
        }
        return "\"" + jsonEscape(value.toString()) + "\"";
    }

    private static String jsonEscape(String value) {
        return value.replace("\\", "\\\\").replace("\"", "\\\"")
                .replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t");
    }

    private static Map<String, String> parse(String[] args) {
        Map<String, String> options = new LinkedHashMap<>();
        for (int i = 0; i < args.length; i++) {
            String token = args[i];
            if (!token.startsWith("--")) throw new IllegalArgumentException("Expected --option, got " + token);
            String key = token.substring(2).replace('-', '_');
            if (i + 1 >= args.length || args[i + 1].startsWith("--")) {
                throw new IllegalArgumentException("Missing value for --" + key);
            }
            options.put(key, args[++i]);
        }
        for (String required : List.of("dataset_idx", "algo")) {
            if (!options.containsKey(required)) throw new IllegalArgumentException("Required option --" + required);
        }
        if (integer(options, "dataset_idx", -1) < 1 || integer(options, "run", 1) < 1) {
            throw new IllegalArgumentException("dataset_idx and run must be positive integers.");
        }
        return options;
    }

    private static int integer(Map<String, String> options, String key, int fallback) {
        try {
            return Integer.parseInt(options.getOrDefault(key, Integer.toString(fallback)));
        } catch (NumberFormatException ex) {
            throw new IllegalArgumentException("Invalid integer for --" + key + ": " + options.get(key));
        }
    }

    private static double decimal(Map<String, String> options, String key, double fallback) {
        try {
            return Double.parseDouble(options.getOrDefault(key, Double.toString(fallback)));
        } catch (NumberFormatException ex) {
            throw new IllegalArgumentException("Invalid number for --" + key + ": " + options.get(key));
        }
    }
}
