﻿using System;
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

        // Your existing code starts here
        // Specify the path to the CSV file
        string inputFilePath = @"./train_20230201.csv";
        
        // Load the dataset and save it as a matrix
        var (observations, targets) = LoadCsvAsMatrix(inputFilePath);

        // Print some basic statistics
        Console.WriteLine("Data Loaded:");
        Console.WriteLine($"Number of observations: {observations.RowCount}");
        Console.WriteLine($"Number of features: {observations.ColumnCount}");
        Console.WriteLine($"Number of targets: {targets.Length}");

        // Perform cross-validation
        int k = 5; // Number of folds
        var crossValidationError = CrossValidate(observations, targets, k);

        // Print cross-validation results
        Console.WriteLine($"Cross-validation total error: {crossValidationError}");

        // Split the dataset into training and testing sets
        var split = SplitData(observations, targets, 0.7);
        var trainingObservations = split.TrainingObservations;
        var trainingTargets = split.TrainingTargets;
        var testingObservations = split.TestingObservations;
        var testingTargets = split.TestingTargets;

        // Initialize and train the AdaBoost model
        var decisionTreeLearner = new ClassificationDecisionTreeLearner(maximumTreeDepth: 1);
        var adaBoostLearner = new ClassificationAdaBoostLearner(
            iterations: 50,
            learningRate: 1.0,
            maximumTreeDepth: 1
            );
        var model = adaBoostLearner.Learn(trainingObservations, trainingTargets);

        // Make predictions
        var predictions = model.Predict(testingObservations);

        // Evaluate the model
        var metric = new TotalErrorClassificationMetric<double>();
        var accuracy = 1.0 - metric.Error(testingTargets, predictions);

        Console.WriteLine($"Accuracy: {accuracy}");

        // Calculate the confusion matrix
        var confusionMatrix = CalculateConfusionMatrix(testingTargets, predictions);
        PrintConfusionMatrix(confusionMatrix);

        // Stop the stopwatch and elapsed time update thread
        stopwatch.Stop();
        _keepRunning = false;
        elapsedTimeThread.Join();  // Wait for the elapsed time thread to finish

        // Print final elapsed time
        Console.WriteLine($"Total Elapsed Time: {stopwatch.Elapsed}");
        // Your existing code ends here
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

static (F64Matrix, double[]) LoadCsvAsMatrix(string inputFilePath)
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
                    double[] observation = new double[numColumns - 2]; // Adjusted for 'SYMOBL' and 'Date' columns

                    for (int i = 2; i < numColumns - 1; i++) // Skip 'SYMOBL' and 'Date'
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

                    if (double.TryParse(fields[numColumns - 1], out double target))
                    {
                        targetsList.Add(target);
                    }
                    else
                    {
                        targetsList.Add(double.NaN); // Handle non-numeric target
                    }
                }

                // Clean data: Remove rows with NaN values
                var cleanObservationsList = new List<double[]>();
                var cleanTargetsList = new List<double>();

                for (int i = 0; i < observationsList.Count; i++)
                {
                    if (!observationsList[i].Any(double.IsNaN) && !double.IsNaN(targetsList[i]))
                    {
                        cleanObservationsList.Add(observationsList[i]);
                        cleanTargetsList.Add(targetsList[i]);
                    }
                }

                // Convert lists to matrix and array
                var observations = new F64Matrix(cleanObservationsList.Count, numColumns - 2);
                for (int i = 0; i < cleanObservationsList.Count; i++)
                {
                    for (int j = 0; j < numColumns - 2; j++)
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

    static (F64Matrix TrainingObservations, double[] TrainingTargets, F64Matrix TestingObservations, double[] TestingTargets) SplitData(F64Matrix observations, double[] targets, double trainRatio)
    {
        int totalRows = observations.RowCount;
        int trainCount = (int)(totalRows * trainRatio);

        var rand = new Random(42);
        var indices = Enumerable.Range(0, totalRows).OrderBy(x => rand.Next()).ToArray();

        var trainingObservations = new F64Matrix(trainCount, observations.ColumnCount);
        var testingObservations = new F64Matrix(totalRows - trainCount, observations.ColumnCount);

        var trainingTargets = new double[trainCount];
        var testingTargets = new double[totalRows - trainCount];

        for (int i = 0; i < trainCount; i++)
        {
            for (int j = 0; j < observations.ColumnCount; j++)
            {
                trainingObservations[i, j] = observations[indices[i], j];
            }
            trainingTargets[i] = targets[indices[i]];
        }

        for (int i = trainCount; i < totalRows; i++)
        {
            for (int j = 0; j < observations.ColumnCount; j++)
            {
                testingObservations[i - trainCount, j] = observations[indices[i], j];
            }
            testingTargets[i - trainCount] = targets[indices[i]];
        }

        return (trainingObservations, trainingTargets, testingObservations, testingTargets);
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

        Console.WriteLine($"Fold {i+1}/{k}:");
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

        // Validate test observations before predicting
        for (int j = 0; j < testObservations.RowCount; j++)
        {
            var observation = new double[testObservations.ColumnCount];
            for (int col = 0; col < testObservations.ColumnCount; col++)
            {
                observation[col] = testObservations[j, col];
            }

            if (observation.All(double.IsNaN))
            {
                Console.WriteLine($"Empty observation at index {j}. Skipping this observation.");
                continue; // Skip empty observations
            }

            try
            {
                var prediction = model.Predict(observation); // Test prediction to catch potential errors
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Prediction error for observation at index {j}: {ex.Message}. Skipping this observation.");
                continue; // Skip observations causing errors
            }
        }

        var predictions = model.Predict(testObservations);

        var error = metric.Error(testTargets, predictions);
        errors.Add(error);
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
