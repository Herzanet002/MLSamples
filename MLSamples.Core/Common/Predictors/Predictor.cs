using Microsoft.ML;

namespace MLSamples.Core.Common.Predictors;

public class Predictor<TModelIn, TModelOut> : IPredictor<TModelIn, TModelOut>
	where TModelIn : class
	where TModelOut : class, new()
{
	private readonly MLContext _mlContext;
	private readonly string _modelPath;

	private ITransformer _model;

	public Predictor(string modelPath)
	{
		_modelPath = modelPath;
		_mlContext = new MLContext(0);
	}

	public TModelOut Predict(TModelIn newSample)
	{
		LoadModel();

		var predictionEngine = _mlContext.Model.CreatePredictionEngine<TModelIn, TModelOut>(_model);

		return predictionEngine.Predict(newSample);
	}

	public void LoadModel()
	{
		if (!File.Exists(_modelPath)) throw new FileNotFoundException($"File {_modelPath} doesn't exist.");

		using (var stream = new FileStream(_modelPath,
			       FileMode.Open,
			       FileAccess.Read,
			       FileShare.Read))
		{
			_model = _mlContext.Model.Load(stream, out _);
		}

		if (_model == null) throw new Exception("Failed to load Model");
	}
}