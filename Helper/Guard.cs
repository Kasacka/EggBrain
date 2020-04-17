using System;
using System.Collections.Generic;
using System.Linq;

namespace EggBrain
{
    internal static class Guard
    {
        public static void NotNull(object parameter, string parameterName)
        {
            if (parameter == null)
                throw new ArgumentNullException(parameterName);
        }

        public static void NotNegative(int value, string parameterName)
        {
            if (value < 0)
                throw new ArgumentOutOfRangeException(parameterName, value, $"{StringHelper.UpperCaseFirst(parameterName)} must not ne negative");
        }

        public static void NotNegative(double value, string parameterName)
        {
            if (value < 0)
                throw new ArgumentOutOfRangeException(parameterName, value, $"{StringHelper.UpperCaseFirst(parameterName)} must not ne negative");
        }

        public static void Maximum(int value, int maxValue, string parameterName)
        {
            if (value > maxValue)
                throw new ArgumentOutOfRangeException(parameterName, value, $"{StringHelper.UpperCaseFirst(parameterName)} must not be greater than {maxValue}");
        }

        public static void Maximum(double value, double maxValue, string parameterName)
        {
            if (value > maxValue)
                throw new ArgumentOutOfRangeException(parameterName, value, $"{StringHelper.UpperCaseFirst(parameterName)} must not be greater than {maxValue}");
        }

        public static void HasLength<T>(IEnumerable<T> enumerable, int length, string message)
        {
            if (enumerable.Count() != length)
                throw new ArgumentException(message);
        }
    }
}