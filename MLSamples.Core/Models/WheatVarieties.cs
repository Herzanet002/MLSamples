namespace MLSamples.Core.Models
{
	public static class WheatVarieties
	{
		public static Dictionary<uint, string> Varieties = new()
		{
			{1, "Камская"},
			{2, "Розовая"},
			{3, "Канадская"}
		};

		public static string? GetVarietyByCluster(uint clusterId)
		{
			Varieties.TryGetValue(clusterId, out var variety);
			return variety;
		}
	}
}
