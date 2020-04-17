namespace EggBrain
{
    internal static class StringHelper
    {
        public static string UpperCaseFirst(string value)
        {
            if (string.IsNullOrEmpty(value))
                return value;
            return char.ToUpper(value[0]) + value.Substring(1);
        }
    }
}