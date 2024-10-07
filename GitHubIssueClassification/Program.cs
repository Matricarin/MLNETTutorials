using Microsoft.ML;
using MLTutorialsModels;

namespace GitHubIssueClassification
{
    internal class Program
    {
        static string _appPath = Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]) ?? ".";
        static string _trainDataPath = Path.Combine(_appPath, "..", "..", "..", "Data", "issues_train.tsv");
        static string _testDataPath = Path.Combine(_appPath, "..", "..", "..", "Data", "issues_test.tsv");
        static string _modelPath = Path.Combine(_appPath, "..", "..", "..", "Models", "model.zip");

        static MLContext _mlContext;
        PredictionEngine<GitHubIssue, IssuePrediction> _predEngine;
        ITransformer _trainedModel;
        IDataView _trainingDataView;

        static void Main(string[] args)
        {
            Console.WriteLine("Hello, ML!");
            _mlContext = new MLContext(seed: 0);
            _mlContext.Data.LoadFromTextFile<GitHubIssue>(_trainDataPath, hasHeader: true);
            var pipeline = ProcessData();

        }

        static IEstimator<ITransformer> ProcessData()
        {

        }
    }
}
