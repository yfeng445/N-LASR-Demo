using System;
using System.Collections.Generic;
using System.IO;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Metrics.Classification;
using SharpLearning.RandomForest.Learners;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers.Matrices;
using SharpLearning.InputOutput.Csv;

class Program
{
    static void Main()
    {
        // Load the dataset
        var parser = new CsvParser(() => new StreamReader(@"path/to/your/BostonHousing.csv"));
        var observations = parser.EnumerateRows("medv").ToF64Matrix();
        var targets = parser.EnumerateRows("medv").ToF64Vector();

        // Create a binary target variable
        double medianValue = Median(targets);
        for (int i = 0; i < targets.Length; i++)
        {
            targets[i] = targets[i] > medianValue ? 1.0 : 0.0;
        }

        // Split the dataset into training and testing sets
        var splitter = new RandomIndexSplitter<double>(0.7, seed: 42);
        var split = splitter.SplitSet(observations, targets);
        var trainingObservations = split.TrainingSet.Observations;
        var trainingTargets = split.TrainingSet.Targets;
        var testingObservations = split.TestingSet.Observations;
        var testingTargets = split.TestingSet.Targets;

        // Initialize and train the AdaBoost model
        var decisionTreeLearner = new ClassificationDecisionTreeLearner(maxTreeDepth: 1);
        var adaBoostLearner = new ClassificationAdaBoostLearner(decisionTreeLearner, iterations: 50);
        var model = adaBoostLearner.Learn(trainingObservations, trainingTargets);

        // Make predictions
        var predictions = model.Predict(testingObservations);

        // Evaluate the model
        var metric = new TotalErrorClassificationMetric<double>();
        var accuracy = 1.0 - metric.Error(testingTargets, predictions);

        Console.WriteLine($"Accuracy: {accuracy}");

        // Confusion matrix and classification report
        var confusionMatrix = new ConfusionMatrix<double>(testingTargets, predictions);
        Console.WriteLine("Confusion Matrix:");
        Console.WriteLine(confusionMatrix);

        Console.WriteLine("Classification Report:");
        Console.WriteLine(confusionMatrix.ToString());
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
}
