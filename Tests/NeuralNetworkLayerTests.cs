using System;
using Xunit;

namespace EggBrain.Tests
{
    public sealed class NeuralNetworkLayerSettingsTests
    {
        [Fact]
        public void Ctor_Test()
        {
            var layer = new NeuralNetworkLayerSettings(2, new SigmoidActivationFunction());
            Assert.Equal(2, layer.NeuronCount);
            Assert.IsType<SigmoidActivationFunction>(layer.ActivationFunction);
        }

        [Fact]
        public void Ctor_NeuronCount_Negative_Test()
        {
            Assert.Throws(typeof(ArgumentOutOfRangeException), () => new NeuralNetworkLayerSettings(-2, new LinearActivationFunction()));
        }

        [Fact]
        public void Ctor_NeuronCount_Overflow_Test()
        {
            Assert.Throws(typeof(ArgumentOutOfRangeException), () => new NeuralNetworkLayerSettings(NeuralNetworkLayerSettings.MaximumNeuronCount + 1, new LinearActivationFunction()));
        }

        [Fact]
        public void Ctor_ActivationFunction_Null_Test()
        {
            Assert.Throws(typeof(ArgumentNullException), () => new NeuralNetworkLayerSettings(2, null));
        }
    }
}