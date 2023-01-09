using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace MLSamples.Core.Common.Trainers;

public abstract class TrainerBase<T> : ITrainerBase where T : class
{
	protected readonly MLContext MlContext;
	protected DataOperationsCatalog.TrainTestData DataSplit;
	protected ITrainerEstimator<SingleFeaturePredictionTransformerBase<T>, T> Model;
	protected ITransformer TrainedModel;

	protected TrainerBase()
	{
		MlContext = new MLContext(0);
	}

	public string Name { get; init; }

	public ITransformer Fit(string trainingFileName)
	{
		if (!File.Exists(trainingFileName)) throw new FileNotFoundException($"File {trainingFileName} doesn't exist.");

		DataSplit = LoadAndPrepareData(trainingFileName);
		var dataProcessPipeline = BuildDataProcessingPipeline();
		var trainingPipeline = dataProcessPipeline.Append(Model);

		TrainedModel = trainingPipeline.Fit(DataSplit.TrainSet);
		return TrainedModel;
	}

	public void Save(string pathToSave)
	{
		MlContext.Model.Save(TrainedModel, DataSplit.TrainSet.Schema, pathToSave);
	}

	protected abstract DataOperationsCatalog.TrainTestData LoadAndPrepareData(string trainingFileName);
	protected abstract IEstimator<ITransformer> BuildDataProcessingPipeline();
}