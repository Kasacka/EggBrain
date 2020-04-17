using System;
using System.Collections.Generic;
using System.Linq;

namespace EggBrain
{
    public sealed class NeuralNetwork
    {
        private readonly NeuralNetworkSettings settings;
        private readonly double[][] layerInputs;
        private readonly double[][] layerOutputs;
        private readonly double[][] deltas;
        private readonly double[][,] synapses;

        public NeuralNetwork(NeuralNetworkSettings settings) 
        {
            Guard.NotNull(settings, nameof(settings));
            this.settings = settings;
            layerInputs = NeuralNetworkArrayFactory.CreateLayers(Layers);
            layerOutputs = NeuralNetworkArrayFactory.CreateLayers(Layers);
            deltas = NeuralNetworkArrayFactory.CreateLayers(Layers);
            synapses = NeuralNetworkArrayFactory.CreateSynapses(Layers);
        }

        public void Train(double[] inputs, double[] expectedOutputs)
        {
            Guard.NotNull(inputs, nameof(inputs));
            Guard.NotNull(expectedOutputs, nameof(expectedOutputs));
            Guard.HasLength(inputs, Layers.First().NeuronCount, "Invalid count of input values");
            Guard.HasLength(expectedOutputs, Layers.Last().NeuronCount, "Invalid count of expected output values");
            Test(inputs);
            PropagateForward();
            PropagateBackward(expectedOutputs);
        }

        public double[] Test(double[] inputs)
        {
            Guard.NotNull(inputs, nameof(inputs));
            Guard.HasLength(inputs, Layers.First().NeuronCount, "Invalid count of input values");
            Array.Copy(inputs, 0, layerInputs.First(), 0, inputs.Length);
            PropagateForward();
            return layerOutputs.Last().ToArray();
        }

        private void PropagateForward()
        {
            for (var layerIndex = 0; layerIndex < LayerCount; ++layerIndex)
                PropagateForward(layerIndex);
        }

        private void PropagateBackward(double[] expectedOutputs)
        {
            for (var layerIndex = LayerCount - 1; layerIndex > 0; --layerIndex)
                PropagateBackward(layerIndex, expectedOutputs);
        }

        private void PropagateForward(int layerIndex)
        {            
            Maths.Map(layerInputs[layerIndex], Layers[layerIndex].ActivationFunction.Perform, layerOutputs[layerIndex]);
            if (layerIndex < LayerCount - 1)
                layerInputs[layerIndex + 1] = Maths.DotProduct(synapses[layerIndex], layerOutputs[layerIndex]);
        }

        private void PropagateBackward(int layerIndex, double[] expectedOutputs)
        {
            var derivations = layerInputs[layerIndex].Select(Layers[layerIndex].ActivationFunction.PerformDerivation).ToArray();
            double[] differences = null;
            if (IsLastLayer(layerIndex))
                differences = GetLastLayerDifferences(expectedOutputs);
            else
                differences = GetHiddenLayerDifferences(layerIndex);
            deltas[layerIndex] = new double[NeuronCount(layerIndex)];
            Maths.Multiply(differences, derivations, deltas[layerIndex]);
            Matrix.Map(synapses[layerIndex - 1], (x, y, current) => current + -deltas[layerIndex][y] * layerOutputs[layerIndex - 1][x] * LearningRate);
        }

        private double[] GetLastLayerDifferences(double[] expectedOutputs)
        {
            var differences = new double[NeuronCount(LayerCount - 1)];
            Maths.Difference(layerOutputs[LayerCount - 1], expectedOutputs, differences);
            return differences;
        }

        private double[] GetHiddenLayerDifferences(int layerIndex) => 
            Enumerable.Range(1, NeuronCount(layerIndex))
                .Select((_, x) => deltas[layerIndex + 1]
                    .Select((delta, y) => delta * synapses[layerIndex][x, y])
                    .Sum()).ToArray();

        private IReadOnlyList<NeuralNetworkLayerSettings> Layers =>
            settings.Layers;

        private int LayerCount =>
            Layers.Count;

        private double LearningRate =>
            settings.LearningRate;

        private int NeuronCount(int layerIndex) => 
            Layers[layerIndex].NeuronCount;

        private bool IsLastLayer(int layerIndex) =>
            layerIndex == LayerCount - 1;
    }
}