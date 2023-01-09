using System.Text;
using MLSamples.Application.Trainers;
using MLSamples.Core.Common.Predictors;
using MLSamples.Core.Models;

var binaryClassificationDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "water_quality.csv");
var clusteringDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "wheat_clustering.csv");

var binaryClassificationModelPath = Path.Combine(Environment.CurrentDirectory, "Data", "WaterQuality.zip");
var clusteringModelPath = Path.Combine(Environment.CurrentDirectory, "Data", "WheatClustering.zip");

static string ReadLineTrimmed()
{
	return Console.ReadLine().TrimStart().TrimEnd();
}

static void DisplayErrorMessage(string errorMessage)
{
	Console.ForegroundColor = ConsoleColor.Red;
	Console.WriteLine($"\n{errorMessage}\n");
	Console.ResetColor();
}

do
{
	Console.WriteLine("Выберите нужный алгоритм обучения:");
	Console.ForegroundColor = ConsoleColor.DarkMagenta;
	Console.WriteLine("1 - Кластеризация с применением алгоритма KMeans:");
	Console.WriteLine("2 - Бинарная классификация с применением случайных деревьев:");
	Console.WriteLine("3 - Выход");
	var chose = ReadLineTrimmed();
	switch (chose)
	{
		case "1":
			Console.ForegroundColor = ConsoleColor.DarkCyan;
			Console.WriteLine("1 - Выбрать тестовые данные для обучения");
			Console.WriteLine("2 - Указать путь к файлу обучения");
			Console.ResetColor();
			var input = ReadLineTrimmed();
			if (input == string.Empty)
			{
				DisplayErrorMessage("Повторите корректный ввод");
				Console.ReadKey();
			}
			else
			{
				RunKMeansTraining();
			}

			break;
		case "2":
			Console.ForegroundColor = ConsoleColor.DarkCyan;
			Console.WriteLine("1 - Выбрать тестовые данные для обучения");
			Console.WriteLine("2 - Указать путь к файлу обучения");
			Console.ResetColor();
			var chosed = ReadLineTrimmed();
			if (chosed == string.Empty)
			{
				DisplayErrorMessage("Повторите корректный ввод");
				Console.ReadKey();
			}
			else
			{
				RunRandomForestTraining();
			}

			break;
		case "3":
			Environment.Exit(0);
			break;
		default:
			DisplayErrorMessage("Повторите корректный ввод");
			break;
	}
} while (true);


void RunRandomForestTraining()
{
	var trainer = new RandomForestTrainer(2, 5);
	Console.WriteLine("*******************************");
	Console.WriteLine($"{trainer.Name}");
	Console.WriteLine("*******************************");

	trainer.Fit(binaryClassificationDataPath);

	var modelMetrics = trainer.Evaluate();
	var sb = new StringBuilder();
	sb.Append($"Accuracy: {modelMetrics.Accuracy:f}\n")
		.Append($"F1 Score: {modelMetrics.F1Score:f}\n")
		.Append($"Positive Precision: {modelMetrics.PositivePrecision:f}\n")
		.Append($"Negative Precision: {modelMetrics.NegativePrecision:f}\n")
		.Append($"Positive Recall: {modelMetrics.PositiveRecall:f}\n")
		.Append($"Negative Recall: {modelMetrics.NegativeRecall:f}\n")
		.Append($"Area Under Precision Recall Curve: {modelMetrics.AreaUnderPrecisionRecallCurve:f}\n");
	Console.WriteLine(sb.ToString());

	trainer.Save(binaryClassificationModelPath);

	var predictor = new Predictor<WaterModel, BinaryPredictionLabel>(binaryClassificationModelPath);
	var prediction = predictor.Predict(new WaterModel
	{
		Aluminium = 1.65f,
		Ammonia = 0.08f,
		Arsenic = 0,
		Barium = 0,
		Cadmium = 0,
		Chloramine = 0,
		Chromium = 0,
		Copper = 0,
		Flouride = 0,
		Bacteria = 0,
		Viruses = 0,
		Lead = 0,
		Nitrates = 0,
		Nitrites = 0,
		Mercury = 0,
		Perchlorate = 0,
		Radium = 0,
		Selenium = 0,
		Silver = 0,
		Uranium = 0
	});
	Console.WriteLine("------------------------------");
	Console.WriteLine($"Prediction: {prediction.PredictedLabel:#.##}");
	Console.WriteLine("------------------------------\n");
}

void RunKMeansTraining()
{
	var trainer = new KMeansTrainer(3);
	Console.WriteLine("*******************************");
	Console.WriteLine($"{trainer.Name}");
	Console.WriteLine("*******************************");

	trainer.Fit(clusteringDataPath);

	var modelMetrics = trainer.Evaluate();

	Console.WriteLine($"AverageDistance: {modelMetrics.AverageDistance:0.##}{Environment.NewLine}" +
	                  $"DaviesBouldinIndex: {modelMetrics.DaviesBouldinIndex:#.##}{Environment.NewLine}" +
	                  $"NormalizedMutualInformation: {modelMetrics.NormalizedMutualInformation:#.##}{Environment.NewLine}");
	trainer.Save(clusteringModelPath);

	var predictor = new Predictor<WheatModel, ClusterPredictionModel>(clusteringModelPath);
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
	Console.WriteLine($"Расстояния: {string.Join(" ", prediction.Distances!)}\n");
}