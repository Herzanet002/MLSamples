using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using MLSamples.Core.Models;

var dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "wheat_clustering.csv");
var modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "IrisClusteringModel.zip");


var mlContext = new MLContext(seed: 0);
var dataView = mlContext.Data.LoadFromTextFile<WheatModel>(dataPath, hasHeader: true, separatorChar: ',');

const string FEATURES_COLUMN_NAME = "Features";
var pipeline = BuildPipeline(mlContext, FEATURES_COLUMN_NAME);

var model = pipeline.Fit(dataView);

using var fileStream = new FileStream(modelPath, FileMode.Create, FileAccess.Write, FileShare.Write);
mlContext.Model.Save(model, dataView.Schema, fileStream);

var predictor = mlContext.Model
	.CreatePredictionEngine<WheatModel, ClusterPredictionModel>(model);

//20.24,16.91,0.8897,6.315,3.962,5.901,6.188,1
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
Console.WriteLine($"Кластер: {prediction.PredictedClusterId}");
Console.WriteLine($"Сорт: {WheatVarieties.GetVarietyByCluster(prediction.PredictedClusterId)}");
Console.WriteLine($"Расстояния: {string.Join(" ", prediction.Distances!)}");


static EstimatorChain<ClusteringPredictionTransformer<KMeansModelParameters>> BuildPipeline(MLContext mlContext,
	string featuresColumnName)
{
	var pipeline = mlContext.Transforms
		.Concatenate(featuresColumnName,
			"Area",
			"Perimeter",
			"Compactness",
			"KernelLength",
			"KernelWidth",
			"AsymmetryCoef",
			"KernelGrooveLength")
		.Append(mlContext.Clustering.Trainers.KMeans(featuresColumnName, numberOfClusters: 3));
	return pipeline;
}