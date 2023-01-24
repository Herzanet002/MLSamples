namespace MLSamples.ConsoleApp.Helpers.Console;

public class ConsoleInputHelper
{
    public static string? ReadLineTrimmed()
    {
        return System.Console.ReadLine()?.TrimStart().TrimEnd();
    }
}