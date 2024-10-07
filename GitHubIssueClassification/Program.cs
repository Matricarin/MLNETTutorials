using Microsoft.ML;

namespace GitHubIssueClassification
{
    internal class Program
    {
        static void Main(string[] args)
        {
            string _appPath = Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]) ?? ".";
            string _trainDataPath = Path.Combine(_appPath, "..", "..", "..", "Data", "issues_train.tsv");
            string _testDataPath = Path.Combine(_appPath, "..", "..", "..", "Data", "issues_test.tsv");
            string _modelPath = Path.Combine(_appPath, "..", "..", "..", "Models", "model.zip");

            MLContext _mlContext;
            PredictionEngine<GitHubIssue, IssuePrediction> _predEngine;
            ITransformer _trainedModel;
            IDataView _trainingDataView;
            Console.WriteLine("Hello, ML!");
        }
    }
}
