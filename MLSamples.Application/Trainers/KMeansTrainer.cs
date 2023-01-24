using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using MLSamples.Core.Common.Trainers;
using MLSamples.Core.Models;

namespace MLSamples.Application.Trainers;

public class KMeansTrainer : TrainerBase<KMeansModelParameters>
{
    public KMeansTrainer(int numberOfClusters)
    {
        Name = $"KMeans: Cluster's number: {numberOfClusters}";
        Model = MlContext.Clustering.Trainers.KMeans(numberOfClusters: numberOfClusters);
    }

    protected override DataOperationsCatalog.TrainTestData LoadAndPrepareData(string trainingFileName)
    {
        var trainingDataView = MlContext.Data
            .LoadFromTextFile<WheatModel>
                (trainingFileName, hasHeader: true, separatorChar: ',');
        return MlContext.Data.TrainTestSplit(trainingDataView, 0.3);
    }

    public ClusteringMetrics Evaluate()
    {
        var testSetTransform = TrainedModel.Transform(DataSplit.TestSet);
        return MlContext.Clustering.Evaluate(testSetTransform);
    }

    protected override EstimatorChain<ClusteringPredictionTransformer<KMeansModelParameters>>
        BuildDataProcessingPipeline()
    {
        var pipeline = MlContext.Transforms
            .Concatenate("Features",
                nameof(WheatModel.Area),
                nameof(WheatModel.Perimeter),
                nameof(WheatModel.Compactness),
                nameof(WheatModel.KernelLength),
                nameof(WheatModel.KernelWidth),
                nameof(WheatModel.AsymmetryCoef),
                nameof(WheatModel.KernelGrooveLength))
            .Append(MlContext.Clustering.Trainers.KMeans(numberOfClusters: 3));
        return pipeline;
    }
}