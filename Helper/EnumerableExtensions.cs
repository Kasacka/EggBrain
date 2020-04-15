using System.Collections.Generic;
using System.Linq;

namespace EggBrain
{
    internal static class EnumerableExtensions
    {
        public static T[,] Reshape<T>(this IEnumerable<T> @this, int width, int height)
        {
            var result = new T[width, height];
            var index = 0;
            foreach (var item in @this)
            {
                var x = index % width;
                var y = index / width;
                result[x, y] = item;
                ++index;
            }
            return result;
        }

        public static T[,] Reshape<T>(this T[][] @this)
        {
            var width = @this.Length;
            var height = @this.First().Length;
            var result = new T[width, height];
            for (var x = 0; x < width; ++x)
                for (var y = 0; y < height; ++y)
                    result[x, y] = @this[x][y];
            return result;
        }
    }
}