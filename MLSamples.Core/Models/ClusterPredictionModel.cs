using Microsoft.ML.Data;

namespace MLSamples.Core.Models;

public sealed class ClusterPredictionModel
{
	[ColumnName("Score")] public float[]? Distances;

	[ColumnName("PredictedLabel")] public uint PredictedClusterId;
}