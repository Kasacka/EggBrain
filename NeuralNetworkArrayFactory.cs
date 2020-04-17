using System;
using System.Collections.Generic;
using System.Linq;

namespace EggBrain
{
    internal static class NeuralNetworkArrayFactory
    {
        private static readonly Random random = new Random();

        public static double[][] CreateLayers(IReadOnlyList<NeuralNetworkLayerSettings> layers) => 
            layers.Select(CreateLayer).ToArray();

        public static double[][,] CreateSynapses(IReadOnlyList<NeuralNetworkLayerSettings> layers) =>
            layers.Skip(1).Select((layer, index) => CreateSynapses(layers[index], layer)).ToArray();

        private static double[,] CreateSynapses(NeuralNetworkLayerSettings inLayer, NeuralNetworkLayerSettings outLayer)
        {
            var result = new double[inLayer.NeuronCount, outLayer.NeuronCount];
            Matrix.Map(result, (x, y) => random.NextDouble());
            return result;
        }

        private static double[] CreateLayer(NeuralNetworkLayerSettings layer) =>
            new double[layer.NeuronCount];
    }
}