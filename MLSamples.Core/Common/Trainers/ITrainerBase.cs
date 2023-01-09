using Microsoft.ML;

namespace MLSamples.Core.Common.Trainers;

public interface ITrainerBase
{
	string Name { get; }
	ITransformer Fit(string trainingFileName);
	void Save(string pathToSave);
}