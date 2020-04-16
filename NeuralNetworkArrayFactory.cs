using System;
using System.Collections.Generic;
using System.Linq;

namespace EggBrain
{
    internal static class NeuralNetworkArrayFactory
    {
        private static readonly Random random = new Random();

        public static double[][] CreateLayers(IReadOnlyList<NeuralNetworkLayer> layers) => 
            layers.Select(CreateLayer).ToArray();

        public static double[][,] CreateSynapses(IReadOnlyList<NeuralNetworkLayer> layers) =>
            layers.Skip(1).Select((layer, index) => CreateSynapses(layers[index], layer)).ToArray();

        private static double[,] CreateSynapses(NeuralNetworkLayer inLayer, NeuralNetworkLayer outLayer)
        {
            var result = new double[inLayer.NeuronCount, outLayer.NeuronCount];
            Matrix.Map(result, (x, y) => random.NextDouble());
            return result;
        }

        private static double[] CreateLayer(NeuralNetworkLayer layer) =>
            new double[layer.NeuronCount];
    }
}