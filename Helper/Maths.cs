using System.Linq;

namespace EggBrain
{
    internal static class Maths
    {
        public static double[] DotProduct(double[,] matrix, double[] vector)
        {
            var result = new double[Height(matrix)];
            for (var y = 0; y < Height(matrix); ++y)
                for (var x = 0; x < Width(matrix); ++x)
                    result[y] += matrix[x, y] * vector[x];
            return result;
        }

        public static void Difference(double[] vector1, double[] vector2, double[] result)
        {
            for (var index = 0; index < vector1.Length; ++index)
                result[index] = vector1[index] - vector2[index];
        }

        public static void Multiply(double[] vector1, double[] vector2, double[] result)
        {
            for (var index = 0; index < vector1.Length; ++index)
                result[index] = vector1[index] * vector2[index];
        }
  
        public static double[] Multiply(double[] vector1, double[] vector2) =>
            vector1.Select((value, index) => value * vector2[index]).ToArray();

        public static int Width(double[,] matrix) =>
            matrix.GetLength(0);

        public static int Height(double[,] matrix) =>
            matrix.GetLength(1);
    }
}