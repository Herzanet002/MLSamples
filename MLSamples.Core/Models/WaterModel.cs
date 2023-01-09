using Microsoft.ML.Data;

namespace MLSamples.Core.Models;

public sealed class WaterModel
{
	[LoadColumn(0)] public float Aluminium { get; set; }

	[LoadColumn(1)] public float Ammonia { get; set; }

	[LoadColumn(2)] public float Arsenic { get; set; }

	[LoadColumn(3)] public float Barium { get; set; }

	[LoadColumn(4)] public float Cadmium { get; set; }

	[LoadColumn(5)] public float Chloramine { get; set; }

	[LoadColumn(6)] public float Chromium { get; set; }

	[LoadColumn(7)] public float Copper { get; set; }

	[LoadColumn(8)] public float Flouride { get; set; }

	[LoadColumn(9)] public float Bacteria { get; set; }

	[LoadColumn(10)] public float Viruses { get; set; }

	[LoadColumn(11)] public float Lead { get; set; }

	[LoadColumn(12)] public float Nitrates { get; set; }

	[LoadColumn(13)] public float Nitrites { get; set; }

	[LoadColumn(14)] public float Mercury { get; set; }

	[LoadColumn(15)] public float Perchlorate { get; set; }

	[LoadColumn(16)] public float Radium { get; set; }

	[LoadColumn(17)] public float Selenium { get; set; }

	[LoadColumn(18)] public float Silver { get; set; }

	[LoadColumn(19)] public float Uranium { get; set; }

	[LoadColumn(20)] public bool Label { get; set; }
}