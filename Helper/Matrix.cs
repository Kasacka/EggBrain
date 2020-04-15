using System;

namespace EggBrain
{
    internal static class Matrix
    {
        public static void Map(double[,] matrix, Func<int, int, double> mapper) => 
            Map(matrix, (x, y, _) => mapper(x, y));

        public static void Map(double[,] matrix, Func<int, int, double, double> mapper)
        {
            for (var x = 0; x < matrix.GetLength(0); ++x)
                for (var y = 0; y < matrix.GetLength(1); ++y)
                    matrix[x, y] = mapper(x, y, matrix[x, y]);
        }
    }
}