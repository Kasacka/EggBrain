using System;

namespace neural
{
    public static class Program
    {
        private static Random random = new Random();

        public static void Main(string[] _)
        {
            var sigmoid = new SigmoidActivationFunction();
            var network = new NeuralNetwork(new[] {
                new NeuralNetworkLayer(2, new LinearActivationFunction()),
                new NeuralNetworkLayer(5, sigmoid),
                new NeuralNetworkLayer(1, sigmoid)
            });
            for (var i = 0; i < 200000; ++i)
                Train(network);
            
            Test(network, false, false);
            Test(network, false, true);
            Test(network, true, false);
            Test(network, true, true);

            //var result = network.Test(new double[] { 1, 1 });
            //Console.WriteLine(string.Join("   ", result));
        }

        private static void Test(NeuralNetwork network, bool x, bool y)
        {
            var res = network.Test(new[] { x ? 1d : 0d, y ? 1d : 0d })[0];
            Console.WriteLine(x + " XOR " + y + " = " + (res > 0.5d));
        }

        private static void Train(NeuralNetwork network)
        {
            var a = random.NextDouble();
            var b = random.NextDouble();
            var x = a < 0.5;
            var y = b < 0.5;
            var res = x ^ y;
            var xv = x ? 1d : 0d;
            var yv = y ? 1d : 0d;
            var resv = res ? 1d : 0d;
            network.Train(new[] { xv, yv }, new[] { resv });
        }
    }
}