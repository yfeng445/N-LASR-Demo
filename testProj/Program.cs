using System;
using System.IO;
using Microsoft.VisualBasic.FileIO;

class Program
{
    static void Main()
    {
        // Specify the path to the CSV file
        string inputFilePath = @"./BostonHousing.csv";
        
        // Load the dataset and print column names
        PrintCsvColumnNames(inputFilePath);
    }

    static void PrintCsvColumnNames(string inputFilePath)
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
                    Console.WriteLine("Column Names in CSV:");
                    foreach (var column in headerFields)
                    {
                        Console.WriteLine(column);
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred: {ex.Message}");
        }
    }
}
