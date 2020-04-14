namespace neural
{
    internal sealed class Matrix
    {
        private readonly double[,] data;

        public Matrix(int width, int height) =>
            data = new double[width, height];

        private Matrix(double[,] data) =>
            this.data = data;

        public int Width => Maths.Width(data);
        public int Height => Maths.Height(data);

        public double this[int width, int height]
        {
            get => data[width, height];
            set => data[width, height] = value;
        }

        public static double[] operator *(Matrix matrix, double[] vector) => 
            Maths.DotProduct(matrix.data, vector);
    }
}