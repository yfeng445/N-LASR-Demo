using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Microsoft.VisualBasic.FileIO;
using SharpLearning.AdaBoost.Learners;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Metrics.Classification;
using SharpLearning.Containers.Matrices;
using SharpLearning.CrossValidation.CrossValidators;
using System.Diagnostics;

class Program
{
    private static bool _keepRunning = true;

    static void Main()
    {
        // Initialize and start stopwatch
        Stopwatch stopwatch = new Stopwatch();
        stopwatch.Start();

        // Start the elapsed time update thread
        Thread elapsedTimeThread = new Thread(() => ShowElapsedTime(stopwatch));
        elapsedTimeThread.Start();

        // Specify the paths to the CSV files
        string trainFilePath = @"./train_20230201.csv";
        string predictFilePath = @"./predict_20230201.csv";

        // Load the training dataset
        var (observations, targets) = LoadCsvAsMatrix(trainFilePath);

        // Print some basic statistics
        Console.WriteLine("Training Data Loaded:");
        Console.WriteLine($"Number of observations: {observations.RowCount}");
        Console.WriteLine($"Number of features: {observations.ColumnCount}");
        Console.WriteLine($"Number of targets: {targets.Length}");

        bool useCrossValidation = false; // Switch to enable or disable cross-validation

        if (useCrossValidation)
        {
            // Perform cross-validation
            int k = 5; // Number of folds
            var crossValidationError = CrossValidate(observations, targets, k);

            // Print cross-validation results
            Console.WriteLine($"\nCross-validation total error: {crossValidationError}");
        }

        // Initialize and train the AdaBoost model
        var decisionTreeLearner = new ClassificationDecisionTreeLearner(maximumTreeDepth: 1);
        var adaBoostLearner = new ClassificationAdaBoostLearner(
            iterations: 50,
            learningRate: 1.0,
            maximumTreeDepth: 1
        );
        var model = adaBoostLearner.Learn(observations, targets);

        // Load the testing dataset
        var (predictObservations, _) = LoadCsvAsMatrix(predictFilePath, false);

        // Make predictions on the testing dataset
        var predictResults = model.Predict(predictObservations);
        Console.WriteLine($"\nPredictions on predict_20230201.csv: {string.Join(", ", predictResults.Take(10))}..."); // Print first 10 predictions

        // Stop the stopwatch and elapsed time update thread
        stopwatch.Stop();
        _keepRunning = false;
        elapsedTimeThread.Join();  // Wait for the elapsed time thread to finish

        // Print final elapsed time
        Console.WriteLine($"Total Elapsed Time: {stopwatch.Elapsed}");
    }

    private static void PrintMatrix(F64Matrix matrix, int step)
    {
        if (step > matrix.RowCount)
        {
            step = matrix.RowCount;
        }
        for (int i = 0; i < step; i++)
        {
            var row = matrix.Row(i);
            Console.WriteLine(i);
            Console.WriteLine(string.Join(", ", row));
        }
    }

    private static void PrintArray(double[] array, int step)
    {
        if (step > array.Length)
        {
            step = array.Length;
        }
        Console.WriteLine("Array length: " + array.Length);
        Console.WriteLine(string.Join(", ", array.Take(step)));
    }

    // Elapsed time update method
    static void ShowElapsedTime(Stopwatch stopwatch)
    {
        while (_keepRunning)
        {
            Console.Write($"\rElapsed Time: {stopwatch.Elapsed}");
            Thread.Sleep(1000);  // Update every second
        }
    }

    private static (F64Matrix, double[]) LoadCsvAsMatrix(string inputFilePath, bool loadTargets = true)
    {
        try
        {
            using (var reader = new StreamReader(inputFilePath))
            {
                using (var parser = new TextFieldParser(reader))
                {
                    parser.TextFieldType = FieldType.Delimited;
                    parser.SetDelimiters(",");

                    // Read header
                    string[] headerFields = parser.ReadFields();
                    int numColumns = headerFields.Length;

                    // Prepare lists to hold the data
                    var observationsList = new List<double[]>();
                    var targetsList = new List<double>();

                    // Read the data
                    while (!parser.EndOfData)
                    {
                        string[] fields = parser.ReadFields();
                        double[] observation = new double[numColumns - (loadTargets ? 3 : 2)]; // Adjusted for 'SYMOBL', 'Date' columns, and the last column if loading targets

                        for (int i = 2; i < numColumns - (loadTargets ? 1 : 0); i++) // Skip 'SYMOBL' and 'Date'
                        {
                            if (double.TryParse(fields[i], out double value))
                            {
                                observation[i - 2] = value;
                            }
                            else
                            {
                                observation[i - 2] = double.NaN; // Handle non-numeric data
                            }
                        }

                        observationsList.Add(observation);

                        if (loadTargets && double.TryParse(fields[numColumns - 1], out double target))
                        {
                            targetsList.Add(target);
                        }
                    }

                    // Clean data: Remove rows with NaN values
                    var cleanObservationsList = new List<double[]>();
                    var cleanTargetsList = new List<double>();

                    for (int i = 0; i < observationsList.Count; i++)
                    {
                        if (!observationsList[i].Any(double.IsNaN) && (!loadTargets || !double.IsNaN(targetsList[i])))
                        {
                            cleanObservationsList.Add(observationsList[i]);
                            if (loadTargets) cleanTargetsList.Add(targetsList[i]);
                        }
                    }

                    // Convert lists to matrix and array
                    var observations = new F64Matrix(cleanObservationsList.Count, numColumns - (loadTargets ? 3 : 2));
                    for (int i = 0; i < cleanObservationsList.Count; i++)
                    {
                        for (int j = 0; j < numColumns - (loadTargets ? 3 : 2); j++)
                        {
                            observations[i, j] = cleanObservationsList[i][j];
                        }
                    }

                    var targets = cleanTargetsList.ToArray();

                    return (observations, targets);
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred: {ex.Message}");
            return (null, null);
        }
    }

    static double Median(double[] values)
    {
        Array.Sort(values);
        int n = values.Length;
        if (n % 2 == 0)
        {
            return (values[n / 2 - 1] + values[n / 2]) / 2.0;
        }
        return values[n / 2];
    }

    static double CrossValidate(F64Matrix observations, double[] targets, int k)
    {
        var metric = new TotalErrorClassificationMetric<double>();
        var errors = new List<double>();
        var random = new Random(42);

        // Generate indices for k-fold cross-validation
        var indices = Enumerable.Range(0, observations.RowCount).OrderBy(x => random.Next()).ToArray();
        var foldSize = observations.RowCount / k;

        for (int i = 0; i < k; i++)
        {
            var testIndices = indices.Skip(i * foldSize).Take(foldSize).ToArray();
            var trainIndices = indices.Except(testIndices).ToArray();

            Console.WriteLine($"\nFold {i + 1}/{k}:");
            Console.WriteLine($"Training set size: {trainIndices.Length}");
            Console.WriteLine($"Testing set size: {testIndices.Length}");

            if (!trainIndices.Any() || !testIndices.Any())
            {
                Console.WriteLine("Empty training or testing set. Skipping this fold.");
                continue; // Skip if no elements in train or test indices
            }

            var trainObservations = CreateMatrix(observations, trainIndices);
            var trainTargets = trainIndices.Select(index => targets[index]).ToArray();
            var testObservations = CreateMatrix(observations, testIndices);
            var testTargets = testIndices.Select(index => targets[index]).ToArray();

            Console.WriteLine($"Train observations row count: {trainObservations.RowCount}");
            Console.WriteLine($"Test observations row count: {testObservations.RowCount}");
            Console.WriteLine($"Train targets count: {trainTargets.Length}");
            Console.WriteLine($"Test targets count: {testTargets.Length}");

            var decisionTreeLearner = new ClassificationDecisionTreeLearner(maximumTreeDepth: 1);
            var adaBoostLearner = new ClassificationAdaBoostLearner(
                iterations: 50,
                learningRate: 1.0,
                maximumTreeDepth: 1
            );

            var model = adaBoostLearner.Learn(trainObservations, trainTargets);

            if (testObservations.RowCount == 0 || testTargets.Length == 0)
            {
                Console.WriteLine("No test observations or test targets. Skipping this fold.");
                continue; // Skip if no elements in test observations or test targets
            }

            try
            {
                var predictions = model.Predict(testObservations);
                var error = metric.Error(testTargets, predictions);
                errors.Add(error);
            }
            catch (InvalidOperationException ex)
            {
                Console.WriteLine($"Prediction error: {ex.Message}. Skipping this fold.");
                continue; // Skip if predictions fail
            }
        }

        if (!errors.Any())
        {
            throw new InvalidOperationException("All folds resulted in empty training or testing sets.");
        }

        return errors.Average();
    }

    static F64Matrix CreateMatrix(F64Matrix original, int[] indices)
    {
        var result = new F64Matrix(indices.Length, original.ColumnCount);
        for (int i = 0; i < indices.Length; i++)
        {
            for (int j = 0; j < original.ColumnCount; j++)
            {
                result[i, j] = original[indices[i], j];
            }
        }
        return result;
    }

    static Dictionary<(double, double), int> CalculateConfusionMatrix(double[] actual, double[] predicted)
    {
        var confusionMatrix = new Dictionary<(double, double), int>();

        for (int i = 0; i < actual.Length; i++)
        {
            var key = (actual[i], predicted[i]);
            if (confusionMatrix.ContainsKey(key))
            {
                confusionMatrix[key]++;
            }
            else
            {
                confusionMatrix[key] = 1;
            }
        }

        return confusionMatrix;
    }

    static void PrintConfusionMatrix(Dictionary<(double, double), int> confusionMatrix)
    {
        Console.WriteLine("Confusion Matrix:");
        foreach (var entry in confusionMatrix)
        {
            Console.WriteLine($"Actual: {entry.Key.Item1}, Predicted: {entry.Key.Item2}, Count: {entry.Value}");
        }
    }
}
