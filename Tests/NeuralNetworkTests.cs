using System;
using System.Linq;
using Xunit;

namespace EggBrain.Tests
{
    public sealed class NeuralNetworkTests
    {
        private static Random random = new Random();

        [Fact]
        public void Ctor_Test()
        {
            // does not throw any exception
            var network = new NeuralNetwork(CreateXorNetworkSettings());
            Assert.NotNull(network);
        }

        [Fact]
        public void Ctor_Settings_Null_Test()
        {
            Assert.Throws(typeof(ArgumentNullException), () => new NeuralNetwork(null));
        }

        [Fact]
        public void Train_Test_Xor_Sample_Test()
        {
            const int iterations = 500000;
            var network = new NeuralNetwork(CreateXorNetworkSettings());
            for (var iteration = 0; iteration < iterations; ++iteration)
                TrainXor(network);
            TestXor(network, false, false, false);
            TestXor(network, false, true, true);
            TestXor(network, true, false, true);
            TestXor(network, true, true, false);
        }

        [Fact]
        public void Train_Inputs_Null_Test()
        {
            var network = new NeuralNetwork(CreateXorNetworkSettings());
            Assert.Throws(typeof(ArgumentNullException), () => network.Train(null, new double[] { 1d }));
        }

        [Fact]
        public void Train_ExpectedOutputs_Null_Test()
        {
            var network = new NeuralNetwork(CreateXorNetworkSettings());
            Assert.Throws(typeof(ArgumentNullException), () => network.Train(null, new double[] { 1d }));
        }

        [Fact]
        public void Train_Inputs_LengthMismatch_Test()
        {
            var network = new NeuralNetwork(CreateXorNetworkSettings());
            Assert.Throws(typeof(ArgumentException), () => network.Train(new double[0], new double[] { 1d }));
        }

        [Fact]
        public void Train_ExpectedOutputs_LengthMismatch_Test()
        {
            var network = new NeuralNetwork(CreateXorNetworkSettings());
            Assert.Throws(typeof(ArgumentException), () => network.Train(new double[] { 1d, 0d }, new double[] { 1d, 2d }));
        }

        [Fact]
        public void Test_Inputs_Null_Test()
        {
            var network = new NeuralNetwork(CreateXorNetworkSettings());
            Assert.Throws(typeof(ArgumentNullException), () => network.Test(null));
        }

        [Fact]
        public void Test_Inputs_LengthMismatch_Test()
        {
            var network = new NeuralNetwork(CreateXorNetworkSettings());
            Assert.Throws(typeof(ArgumentException), () => network.Test(Enumerable.Repeat(2d, 5).ToArray()));
        }

        private static NeuralNetworkSettings CreateXorNetworkSettings()
        {
            return new NeuralNetworkSettings(0.05, new[] {
                new NeuralNetworkLayerSettings(2, new LinearActivationFunction()),
                new NeuralNetworkLayerSettings(7, new SigmoidActivationFunction()),
                new NeuralNetworkLayerSettings(1, new SigmoidActivationFunction())
            });
        }

        private static void TrainXor(NeuralNetwork network)
        {
            var input1 = Math.Round(random.NextDouble());
            var input2 = Math.Round(random.NextDouble());
            var output = ((input1 > 0.5) ^ (input2 > 0.5)) ? 1d : 0d;
            network.Train(new[] { input1, input2 }, new[] { output });
        }

        private static void TestXor(NeuralNetwork network, bool input1, bool input2, bool expectedOutput)
        {
            var result = network.Test(new double[] { input1 ? 1d : 0d, input2 ? 1d : 0d });
            Assert.Equal(1, result.Length);
            var value = result[0];
            Assert.Equal(expectedOutput, value > 0.5d);
        }
    }
}