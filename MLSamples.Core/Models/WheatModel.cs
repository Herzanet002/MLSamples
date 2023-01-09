using Microsoft.ML.Data;

namespace MLSamples.Core.Models;

public sealed class WheatModel
{
	[LoadColumn(0)] public float Area;

	[LoadColumn(5)] public float AsymmetryCoef;

	[LoadColumn(2)] public float Compactness;

	[LoadColumn(6)] public float KernelGrooveLength;

	[LoadColumn(3)] public float KernelLength;

	[LoadColumn(4)] public float KernelWidth;

	[LoadColumn(1)] public float Perimeter;
}