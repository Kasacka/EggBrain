using System.Linq;

namespace neural
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

        public static double[] Difference(double[] vector1, double[] vector2) =>
            vector1.Select((value, index) => value - vector2[index]).ToArray();
  
        public static double[] Multiply(double[] vector1, double[] vector2) =>
            vector1.Select((value, index) => value * vector2[index]).ToArray();

        public static int Width(double[,] matrix) =>
            matrix.GetLength(0);

        public static int Height(double[,] matrix) =>
            matrix.GetLength(1);
    }
}