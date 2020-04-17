using System;
using Xunit;

namespace EggBrain.Tests
{
    public sealed class NeuralNetworkSettingsTests
    {
        [Fact]
        public void Ctor_Test()
        {
            var settings = new NeuralNetworkSettings(0.05, CreateLayerSettings());
            Assert.Equal(0.05, settings.LearningRate);
            Assert.IsType<NeuralNetworkLayerSettings[]>(settings.Layers);
        }

        [Fact]
        public void Ctor_LearningRate_Negative_Test()
        {
            Assert.Throws(typeof(ArgumentOutOfRangeException), () => new NeuralNetworkSettings(-2, CreateLayerSettings()));
        }

        [Fact]
        public void Ctor_LearningRate_Overflow_Test()
        {
            Assert.Throws(typeof(ArgumentOutOfRangeException), () => new NeuralNetworkSettings(70, CreateLayerSettings()));
        }

        [Fact]
        public void Ctor_LayerSettings_Null_Test()
        {
            Assert.Throws(typeof(ArgumentNullException), () => new NeuralNetworkSettings(0.5, null));
        }

        [Fact]
        public void Ctor_LayerSettings_NoLayers_Test()
        {
            Assert.Throws(typeof(ArgumentException), () => new NeuralNetworkSettings(0.5, new NeuralNetworkLayerSettings[0]));
        }

        private static NeuralNetworkLayerSettings[] CreateLayerSettings() => 
            new[] { new NeuralNetworkLayerSettings(5, new SigmoidActivationFunction()) };
    }
}