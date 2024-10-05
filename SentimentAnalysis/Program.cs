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

    private static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
    {
        Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
        IDataView predictions = model.Transform(splitTestSet);
        CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
        Console.WriteLine();
        Console.WriteLine("Model quality metrics evaluation");
        Console.WriteLine("--------------------------------");
        Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
        Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
        Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
        Console.WriteLine("=============== End of model evaluation ===============");
    }

    static void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
    {
        PredictionEngine<SentimentData, SentimentPrediction> predictionFunction =
            mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
        SentimentData sampleStatement = new SentimentData()
        {
            SentimentText = "This was a very bad steak"
        };
        var resultPrediction = predictionFunction.Predict(sampleStatement);
        Console.WriteLine();
        Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

        Console.WriteLine();
        Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");

        Console.WriteLine("=============== End of Predictions ===============");
        Console.WriteLine();
    }
}