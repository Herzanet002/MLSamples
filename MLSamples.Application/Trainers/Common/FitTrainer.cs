using MLSamples.Core.Common.Trainers;

namespace MLSamples.Application.Trainers.Common;

public class FitTrainer<T> : IFitTrainer<T> where T : ITrainerBase
{
    private readonly KMeansTrainer _kMeansTrainer;

    public FitTrainer(KMeansTrainer kMeansTrainer)
    {
        _kMeansTrainer = kMeansTrainer;
    }

    public IFitTrainer<T> Fit(string trainingFileName)
    {
        _kMeansTrainer.Fit(trainingFileName);
        return this;
    }
}

public interface IFitTrainer<T> where T : ITrainerBase
{
}