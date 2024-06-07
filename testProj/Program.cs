using System;
using System.IO;
using Microsoft.VisualBasic.FileIO;
using SharpLearning.Containers.Matrices;

class Program
{
    static void Main()
    {
        // Specify the path to the CSV file
        string inputFilePath = @"./BostonHousing.csv";
        
        // Load the dataset and save it as a matrix
        var (observations, targets) = LoadCsvAsMatrix(inputFilePath);

        // Print some basic statistics
        Console.WriteLine("Data Loaded:");
        Console.WriteLine($"Number of observations: {observations.RowCount}");
        Console.WriteLine($"Number of features: {observations.ColumnCount}");
        Console.WriteLine($"Number of targets: {targets.Length}");
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
                        double[] observation = new double[numColumns - 1];

                        for (int i = 0; i < numColumns - 1; i++)
                        {
                            observation[i] = double.Parse(fields[i]);
                        }

                        observationsList.Add(observation);
                        targetsList.Add(double.Parse(fields[numColumns - 1]));
                    }

                    // Convert lists to matrix and array
                    var observations = new F64Matrix(observationsList.Count, numColumns - 1);
                    for (int i = 0; i < observationsList.Count; i++)
                    {
                        for (int j = 0; j < numColumns - 1; j++)
                        {
                            observations[i, j] = observationsList[i][j];
                        }
                    }

                    var targets = targetsList.ToArray();

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
}
