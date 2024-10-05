using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;

namespace SentimentAnalysis
{
    internal class Program
    {
        private static string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, ML!");
            MLContext mlContext = new MLContext();
            TrainTestData splitDataView = LoadData(mlContext);
        }

        static TrainTestData LoadData(MLContext mlContext)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);
            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            return splitDataView;
        }
    }
}
