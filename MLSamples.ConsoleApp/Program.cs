using System.Text;
using MLSamples.Application.Trainers;
using MLSamples.ConsoleApp.Helpers.Console;
using MLSamples.Core.Common.Predictors;
using MLSamples.Core.Models;

var binaryClassificationDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "water_quality.csv");
var clusteringDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "wheat_clustering.csv");

var binaryClassificationModelPath = Path.Combine(Environment.CurrentDirectory, "Data", "WaterQuality.zip");
var clusteringModelPath = Path.Combine(Environment.CurrentDirectory, "Data", "WheatClustering.zip");

do
{
    ConsoleMessageHelper.DisplayWelcomeMessage();
    var chose = ConsoleInputHelper.ReadLineTrimmed();
    ResolveLearningMethodByMode(chose);
} while (true);

void ResolveLearningMethodByMode(string? mode)
{
    switch (mode)
    {
        case "1":
            RunKMeansTraining();
            break;

        case "2":
            RunRandomForestTraining();
            break;
        case "3":
            Environment.Exit(0);
            break;
        default:
            ConsoleMessageHelper.DisplayErrorMessage("Повторите корректный ввод");
            break;
    }
}

void RunRandomForestTraining()
{
    var trainers = new List<RandomForestTrainer>
    {
        new(2, 5),
        new(3, 6),
        new(12, 15)
    };
    var sample = new WaterModel
    {
        Aluminium = 1.65f,
        Ammonia = 0.08f,
        Arsenic = 0.05f
    };
    trainers.ForEach(t => EvaluatePredict(t, sample));

    void EvaluatePredict(RandomForestTrainer trainer, WaterModel testSample)
    {
        var outputBuilder = new StringBuilder();
        outputBuilder.AppendLine("---------------------------------------------")
            .AppendLine($"{trainer.Name}")
            .AppendLine("---------------------------------------------");
        Console.WriteLine(outputBuilder);
        outputBuilder.Clear();

        trainer.Fit(binaryClassificationDataPath);

        var modelMetrics = trainer.Evaluate();

        outputBuilder.AppendLine($"Accuracy: {modelMetrics.Accuracy:f}")
            .AppendLine($"F1 Score: {modelMetrics.F1Score:f}")
            .AppendLine($"Positive Precision: {modelMetrics.PositivePrecision:f}")
            .AppendLine($"Negative Precision: {modelMetrics.NegativePrecision:f}")
            .AppendLine($"Positive Recall: {modelMetrics.PositiveRecall:f}")
            .AppendLine($"Negative Recall: {modelMetrics.NegativeRecall:f}")
            .AppendLine($"Area Under Precision Recall Curve: {modelMetrics.AreaUnderPrecisionRecallCurve:f}");
        Console.WriteLine(outputBuilder);
        outputBuilder.Clear();

        trainer.Save(binaryClassificationModelPath);

        var predictor = new Predictor<WaterModel, BinaryPredictionLabel>(binaryClassificationModelPath);
        var prediction = predictor.Predict(testSample);

        outputBuilder.AppendLine("------------------------------")
            .AppendLine($"Prediction: {prediction.PredictedLabel}")
            .AppendLine("------------------------------");
        Console.WriteLine(outputBuilder);
    }
}

void RunKMeansTraining()
{
    var trainer = new KMeansTrainer(3);
    var outputBuilder = new StringBuilder();
    outputBuilder.AppendLine("*******************************")
        .AppendLine($"{trainer.Name}")
        .AppendLine("*******************************");
    Console.WriteLine(outputBuilder);
    outputBuilder.Clear();

    trainer.Fit(clusteringDataPath);

    var modelMetrics = trainer.Evaluate();

    outputBuilder.AppendLine($"AverageDistance: {modelMetrics.AverageDistance:f}")
        .AppendLine($"DaviesBouldinIndex: {modelMetrics.DaviesBouldinIndex:f}")
        .AppendLine($"NormalizedMutualInformation: {modelMetrics.NormalizedMutualInformation:f}");

    Console.WriteLine(outputBuilder);
    trainer.Save(clusteringModelPath);

    var predictor = new Predictor<WheatModel, ClusterPredictionModel>(clusteringModelPath);
    //20.24,16.91,0.8897,6.315,3.962,5.901,6.188,=1
    var pink = new WheatModel
    {
        Area = 20.2f,
        Perimeter = 17,
        Compactness = 0.9f,
        KernelLength = 6.3f,
        KernelWidth = 4f,
        AsymmetryCoef = 5.9f,
        KernelGrooveLength = 6f
    };
    var prediction = predictor.Predict(pink);
    outputBuilder.Clear();
    outputBuilder.AppendLine($"Кластер: {prediction.PredictedClusterId}")
        .AppendLine($"Сорт: {WheatVarieties.GetVarietyByCluster(prediction.PredictedClusterId)}")
        .AppendLine($"Расстояния: {string.Join(" ", prediction.Distances!)}");
    Console.WriteLine(outputBuilder);
}