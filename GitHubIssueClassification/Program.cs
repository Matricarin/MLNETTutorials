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
        static PredictionEngine<GitHubIssue, IssuePrediction> _predEngine;
        static ITransformer _trainedModel;
        static IDataView _trainingDataView;

        static void Main(string[] args)
        {
            Console.WriteLine("Hello, ML!");
            _mlContext = new MLContext(seed: 0);
            _trainingDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_trainDataPath, hasHeader: true);
            var pipeline = ProcessData();
            var trainingPipeline = BuildAndTrainModel(_trainingDataView, pipeline);
        }

        static IEstimator<ITransformer> ProcessData()
        {
            var pipeline =
                _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
                    .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title",
                        outputColumnName: "TitleFeaturized"))
                    .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Description",
                        outputColumnName: "DescriptionFeaturized"))
                    .Append(_mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
                    .AppendCacheCheckpoint(_mlContext);
            return pipeline;
        }

        static IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
        {
            var trainingPipeline = pipeline
                .Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            _trainedModel = trainingPipeline.Fit(trainingDataView);
            _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(_trainedModel);
            GitHubIssue issue = new GitHubIssue()
            {
                Title = "WebSockets communication is slow in my machine",
                Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
            };
            var prediction = _predEngine.Predict(issue);
            Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Area} ===============");
            return trainingPipeline;
        }

        void Evaluate(DataViewSchema trainingDataViewSchema)
        {

        }
    }
}
