using Microsoft.ML;
using static Microsoft.ML.DataOperationsCatalog;

namespace SentimentAnalysis;

internal class Program
{
    private static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");

    private static void Main(string[] args)
    {
        Console.WriteLine("Hello, ML!");
        var mlContext = new MLContext();
        TrainTestData splitDataView = LoadData(mlContext);
        ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);
    }

    private static TrainTestData LoadData(MLContext mlContext)
    {
        var dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);
        var splitDataView = mlContext.Data.TrainTestSplit(dataView, 0.2);
        return splitDataView;
    }

    private static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
    {
        var estimator = mlContext.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.SentimentText))
            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression("Label", "Features"));
        Console.WriteLine("=============== Create and Train the Model ===============");
        ITransformer model = estimator.Fit(splitTrainSet);
        Console.WriteLine("=============== End of training ===============");
        Console.WriteLine();
        return model;
    }

    private void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
    {
    }
}