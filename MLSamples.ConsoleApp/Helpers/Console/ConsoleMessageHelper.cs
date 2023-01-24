namespace MLSamples.ConsoleApp.Helpers.Console;

public class ConsoleMessageHelper
{
    public static void DisplayWelcomeMessage()
    {
        System.Console.WriteLine("Выберите нужный алгоритм обучения:");
        System.Console.ForegroundColor = ConsoleColor.DarkMagenta;
        System.Console.WriteLine("1 - Кластеризация с применением алгоритма KMeans:");
        System.Console.WriteLine("2 - Бинарная классификация с применением случайных деревьев:");
        System.Console.WriteLine("3 - Выход");
    }

    public static void DisplayErrorMessage(string errorMessage)
    {
        System.Console.ForegroundColor = ConsoleColor.Red;
        System.Console.WriteLine($"\n{errorMessage}\n");
        System.Console.ResetColor();
    }
}