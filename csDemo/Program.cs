using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.VisualBasic.FileIO;
using SharpLearning.AdaBoost.Learners;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Metrics.Classification;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers.Matrices;
using SharpLearning.InputOutput.Csv;

class Program
{
    static void Main()
    {
        // Preprocess the CSV file to ensure correct formatting
        string inputFilePath = @"./BostonHousing.csv";
        string outputFilePath = @"./PreprocessedBostonHousing.csv";
        PreprocessCsv(inputFilePath, outputFilePath);

        // Load the preprocessed dataset and print column names
        var parser = new CsvParser(() => new StreamReader(outputFilePath));
        var header = parser.EnumerateRows().First().ColumnNameToIndex.Keys;
        Console.WriteLine("Column Names in CSV:");
        foreach (var column in header)
        {
            Console.WriteLine(column);
        }

        // Check if the column count is correct
        if (header.Count() <= 1)
        {
            Console.WriteLine("The CSV file does not contain the correct number of columns.TTTTTTTTTTTTTTTTT");
            return;
        }
        else{
            Console.WriteLine("There are", header.Count(), "columns in total");
        }

        // Load the dataset
        var observations = parser.EnumerateRows("medv").ToF64Matrix();
        var targets = parser.EnumerateRows("medv").ToF64Vector();

        // Rest of your code...
        // Create a binary target variable
        double medianValue = Median(targets);
        for (int i = 0; i < targets.Length; i++)
        {
            targets[i] = targets[i] > medianValue ? 1.0 : 0.0;
        }

        // Split the dataset into training and testing sets
        var split = SplitData(observations, targets, 0.7);
        var trainingObservations = split.TrainingObservations;
        var trainingTargets = split.TrainingTargets;
        var testingObservations = split.TestingObservations;
        var testingTargets = split.TestingTargets;

        // Initialize and train the AdaBoost model
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
    }

    static void PreprocessCsv(string inputFilePath, string outputFilePath)
    {
        using (var reader = new StreamReader(inputFilePath))
        using (var writer = new StreamWriter(outputFilePath))
        {
            // Use TextFieldParser to handle CSV reading properly
            using (var parser = new TextFieldParser(reader))
            {
                parser.TextFieldType = FieldType.Delimited;
                parser.SetDelimiters(",");

                // Read header
                string[] headerFields = parser.ReadFields();
                writer.WriteLine(string.Join(",", headerFields));

                // Read and write the rest of the file
                while (!parser.EndOfData)
                {
                    string[] fields = parser.ReadFields();
                    writer.WriteLine(string.Join(",", fields));
                }
            }
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
