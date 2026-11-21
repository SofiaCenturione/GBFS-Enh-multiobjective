package org.gbfs.math;

public final class MatrixOps {
    private MatrixOps() {}

    public static double[][] minMaxZeroOne(double[][] values) {
        validateRectangular(values);
        int columns = values[0].length;
        double[] minimum = new double[columns];
        double[] maximum = new double[columns];
        for (int column = 0; column < columns; column++) {
            minimum[column] = Double.POSITIVE_INFINITY;
            maximum[column] = Double.NEGATIVE_INFINITY;
            for (double[] row : values) {
                minimum[column] = Math.min(minimum[column], row[column]);
                maximum[column] = Math.max(maximum[column], row[column]);
            }
        }

        double[][] normalized = new double[values.length][columns];
        for (int row = 0; row < values.length; row++) {
            for (int column = 0; column < columns; column++) {
                double range = maximum[column] - minimum[column];
                if (!Double.isFinite(range)) {
                    throw new IllegalArgumentException("Matrix contains non-finite values.");
                }
                normalized[row][column] = range == 0.0
                        ? 0.0
                        : (values[row][column] - minimum[column]) / range;
            }
        }
        return normalized;
    }

    public static double[][] absoluteCorrelation(double[][] samplesByFeature) {
        validateRectangular(samplesByFeature);
        int sampleCount = samplesByFeature.length;
        int featureCount = samplesByFeature[0].length;
        double[][] result = new double[featureCount][featureCount];
        double[] means = new double[featureCount];

        for (int feature = 0; feature < featureCount; feature++) {
            for (double[] sample : samplesByFeature) {
                means[feature] += sample[feature];
            }
            means[feature] /= sampleCount;
        }

        for (int left = 0; left < featureCount; left++) {
            for (int right = left; right < featureCount; right++) {
                double covariance = 0.0;
                double leftVariance = 0.0;
                double rightVariance = 0.0;
                for (double[] sample : samplesByFeature) {
                    double leftDelta = sample[left] - means[left];
                    double rightDelta = sample[right] - means[right];
                    covariance += leftDelta * rightDelta;
                    leftVariance += leftDelta * leftDelta;
                    rightVariance += rightDelta * rightDelta;
                }
                double denominator = Math.sqrt(leftVariance * rightVariance);
                double correlation = denominator == 0.0 ? 0.0 : covariance / denominator;
                correlation = Math.abs(correlation);
                result[left][right] = correlation;
                result[right][left] = correlation;
            }
        }
        return result;
    }

    public static void validateRectangular(double[][] values) {
        if (values == null || values.length == 0 || values[0] == null || values[0].length == 0) {
            throw new IllegalArgumentException("Matrix must not be empty.");
        }
        int columns = values[0].length;
        for (double[] row : values) {
            if (row == null || row.length != columns) {
                throw new IllegalArgumentException("Matrix must be rectangular.");
            }
        }
    }
}
