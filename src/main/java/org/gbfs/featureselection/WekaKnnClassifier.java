package org.gbfs.featureselection;

import weka.classifiers.lazy.IBk;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.EuclideanDistance;
import weka.core.SelectedTag;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public final class WekaKnnClassifier {
    private WekaKnnClassifier() {}

    public static double accuracy(double[][] train, double[] trainLabels,
                                  double[][] test, double[] testLabels,
                                  int[] selectedFeatures, int k) throws Exception {
        validate(train, trainLabels, test, testLabels, selectedFeatures, k);
        if (test.length == 0 || selectedFeatures.length == 0) return 0.0;
        if (k > train.length) {
            throw new IllegalArgumentException("knn_k cannot exceed the number of training samples.");
        }

        List<String> classNames = classNames(trainLabels);
        Map<Double, Integer> classIndices = new TreeMap<>();
        for (int index = 0; index < classNames.size(); index++) {
            classIndices.put(Double.parseDouble(classNames.get(index)), index);
        }

        ArrayList<Attribute> attributes = new ArrayList<>(selectedFeatures.length + 1);
        for (int feature : selectedFeatures) {
            attributes.add(new Attribute("feature_" + feature));
        }
        attributes.add(new Attribute("class", classNames));

        Instances trainingData = new Instances("gbfs_training", attributes, train.length);
        trainingData.setClassIndex(selectedFeatures.length);
        for (int row = 0; row < train.length; row++) {
            double[] values = new double[selectedFeatures.length + 1];
            for (int column = 0; column < selectedFeatures.length; column++) {
                double value = train[row][selectedFeatures[column]];
                values[column] = Double.isFinite(value) ? value : Utils.missingValue();
            }
            Integer classIndex = classIndices.get(trainLabels[row]);
            if (classIndex == null) {
                throw new IllegalArgumentException("Training label is not represented by the nominal class attribute.");
            }
            values[selectedFeatures.length] = classIndex;
            trainingData.add(new DenseInstance(1.0, values));
        }

        EuclideanDistance distance = new EuclideanDistance(trainingData);
        distance.setDontNormalize(true);
        LinearNNSearch search = new LinearNNSearch();
        search.setDistanceFunction(distance);

        IBk classifier = new IBk(k);
        classifier.setCrossValidate(false);
        classifier.setDistanceWeighting(new SelectedTag(IBk.WEIGHT_NONE, IBk.TAGS_WEIGHTING));
        classifier.setNearestNeighbourSearchAlgorithm(search);
        classifier.buildClassifier(trainingData);

        int correct = 0;
        for (int row = 0; row < test.length; row++) {
            double[] values = new double[selectedFeatures.length + 1];
            for (int column = 0; column < selectedFeatures.length; column++) {
                double value = test[row][selectedFeatures[column]];
                values[column] = Double.isFinite(value) ? value : Utils.missingValue();
            }
            values[selectedFeatures.length] = Utils.missingValue();
            Instance instance = new DenseInstance(1.0, values);
            instance.setDataset(trainingData);
            double prediction = classifier.classifyInstance(instance);
            if (Double.isFinite(prediction)
                    && (int) prediction == classIndices.getOrDefault(testLabels[row], -1)) {
                correct++;
            }
        }
        return (double) correct / test.length;
    }

    private static void validate(double[][] train, double[] trainLabels,
                                 double[][] test, double[] testLabels,
                                 int[] selectedFeatures, int k) {
        if (train == null || train.length == 0 || trainLabels == null
                || trainLabels.length != train.length || test == null || testLabels == null
                || testLabels.length != test.length || selectedFeatures == null || k < 1) {
            throw new IllegalArgumentException("Invalid KNN training, test, feature, or neighbor data.");
        }
        int featureCount = train[0].length;
        for (double[] row : train) {
            if (row.length != featureCount) throw new IllegalArgumentException("Training data must be rectangular.");
        }
        for (double[] row : test) {
            if (row.length != featureCount) throw new IllegalArgumentException("Training and test features must align.");
        }
        for (int feature : selectedFeatures) {
            if (feature < 0 || feature >= featureCount) {
                throw new IllegalArgumentException("Selected feature index is outside the data matrix.");
            }
        }
        for (double label : trainLabels) {
            if (!Double.isFinite(label)) throw new IllegalArgumentException("Class labels must be finite.");
        }
        for (double label : testLabels) {
            if (!Double.isFinite(label)) throw new IllegalArgumentException("Class labels must be finite.");
        }
    }

    private static List<String> classNames(double[] labels) {
        TreeMap<Double, Boolean> classes = new TreeMap<>();
        for (double label : labels) classes.put(label, Boolean.TRUE);
        List<String> result = new ArrayList<>(classes.size());
        for (double label : classes.keySet()) result.add(Double.toString(label));
        return result;
    }

}
