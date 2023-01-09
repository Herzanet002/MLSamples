namespace MLSamples.Core.Common.Predictors;

public interface IPredictor<in TModelIn, out TModelOut> where TModelIn : class
	where TModelOut : class, new()
{
	TModelOut Predict(TModelIn newSample);
	void LoadModel();
}