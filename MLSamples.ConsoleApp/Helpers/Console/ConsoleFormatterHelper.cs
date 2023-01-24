namespace MLSamples.ConsoleApp.Helpers.Console;

public class ConsoleFormatterHelper
{
    public static string? ReadLineTrimmed()
    {
        return System.Console.ReadLine()?.TrimStart().TrimEnd();
    }

}