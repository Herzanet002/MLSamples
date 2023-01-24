using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Transforms;
using MLSamples.Core.Common.Trainers;
using MLSamples.Core.Models;

namespace MLSamples.Application.Trainers;

public class RandomForestTrainer : TrainerBase<FastForestBinaryModelParameters>
{
    public RandomForestTrainer(int numberOfLeaves, int numberOfTrees)
    {
        Name = $"Random Forest: Leaves - {numberOfLeaves}, Trees - {numberOfTrees}";
        Model = MlContext.BinaryClassification.Trainers.FastForest(
            numberOfLeaves: numberOfLeaves,
            numberOfTrees: numberOfTrees);
    }

    protected override DataOperationsCatalog.TrainTestData LoadAndPrepareData(string trainingFileName)
    {
        var trainingDataView = MlContext.Data
            .LoadFromTextFile<WaterModel>
                (trainingFileName, hasHeader: true, separatorChar: ',');
        return MlContext.Data.TrainTestSplit(trainingDataView, 0.3);
    }

    public BinaryClassificationMetrics Evaluate()
    {
        var testSetTransform = TrainedModel.Transform(DataSplit.TestSet);
        return MlContext.BinaryClassification.EvaluateNonCalibrated(testSetTransform);
    }

    protected override EstimatorChain<NormalizingTransformer> BuildDataProcessingPipeline()
    {
        var dataProcessPipeline = MlContext.Transforms.Concatenate("Features",
                nameof(WaterModel.Aluminium),
                nameof(WaterModel.Ammonia),
                nameof(WaterModel.Arsenic),
                nameof(WaterModel.Barium),
                nameof(WaterModel.Cadmium),
                nameof(WaterModel.Chloramine),
                nameof(WaterModel.Chromium),
                nameof(WaterModel.Copper),
                nameof(WaterModel.Flouride),
                nameof(WaterModel.Bacteria),
                nameof(WaterModel.Viruses),
                nameof(WaterModel.Lead),
                nameof(WaterModel.Nitrates),
                nameof(WaterModel.Nitrites),
                nameof(WaterModel.Mercury),
                nameof(WaterModel.Perchlorate),
                nameof(WaterModel.Radium),
                nameof(WaterModel.Selenium),
                nameof(WaterModel.Silver),
                nameof(WaterModel.Uranium)
            )
            .Append(MlContext.Transforms.NormalizeMinMax("Features", "Features"))
            .AppendCacheCheckpoint(MlContext);

        return dataProcessPipeline;
    }
}