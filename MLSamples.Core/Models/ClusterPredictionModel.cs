using Microsoft.ML.Data;

namespace MLSamples.Core.Models;

public sealed class ClusterPredictionModel
{
	[ColumnName("PredictedLabel")]
	public uint PredictedClusterId;

	[ColumnName("Score")]
	public float[]? Distances;
}