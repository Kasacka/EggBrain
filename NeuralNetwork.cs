using System;
using System.Collections.Generic;
using System.Linq;

namespace EggBrain
{
    public sealed class NeuralNetwork
    {
        private static readonly Random random = new Random();
        private readonly NeuralNetworkSettings settings;
        private readonly double[][] layerInputs;
        private readonly double[][] layerOutputs;
        private readonly double[][] deltas;
        private readonly double[][,] synapses;

        public NeuralNetwork(NeuralNetworkSettings settings) 
        {
            this.settings = settings;
            layerInputs = CreateLayers();
            layerOutputs = CreateLayers();
            deltas = CreateLayers();
            synapses = CreateSynapses();
        }

        public void Train(double[] inputs, double[] expectedOutputs)
        {
            Test(inputs);
            for (var layerIndex = LayerCount - 1; layerIndex > 0; --layerIndex)
                PropagateBackward(layerIndex, expectedOutputs);
        }

        public double[] Test(double[] inputs)
        {
            Array.Copy(inputs, 0, layerInputs[0], 0, inputs.Length);
            for (var layerIndex = 0; layerIndex < LayerCount; ++layerIndex)
                PropagateForward(layerIndex);
            return layerOutputs[LayerCount - 1].ToArray();
        }

        private void PropagateForward(int layerIndex)
        {            
            Maths.Map(layerInputs[layerIndex], Layers[layerIndex].ActivationFunction.Perform, layerOutputs[layerIndex]);
            if (layerIndex < LayerCount - 1)
                layerInputs[layerIndex + 1] = Maths.DotProduct(synapses[layerIndex], layerOutputs[layerIndex]);
        }

        private void PropagateBackward(int layerIndex, double[] errors)
        {
            var derivations = layerInputs[layerIndex].Select(Layers[layerIndex].ActivationFunction.PerformDerivation).ToArray();
            var differences = new double[NeuronCount(layerIndex)];
            if (IsLastLayer(layerIndex))
                Maths.Difference(layerOutputs[layerIndex], errors, differences);
            else
                NewMethod(layerIndex, differences);

            deltas[layerIndex] = new double[NeuronCount(layerIndex)];
            Maths.Multiply(differences, derivations, deltas[layerIndex]);
            var deltaWeights = new double[NeuronCount(layerIndex - 1), NeuronCount(layerIndex)];
            Matrix.Map(synapses[layerIndex - 1], (x, y, current) => current + -deltas[layerIndex][y] * layerOutputs[layerIndex - 1][x] * LearningRate);
        }

        private void NewMethod(int layerIndex, double[] differences)
        {
            for (var x = 0; x < Layers[layerIndex].NeuronCount; ++x)
            {
                differences[x] = deltas[layerIndex + 1]
                    .Select((h, y) => h * synapses[layerIndex][x, y])
                    .Sum();
            }
        }

        private double[][] CreateLayers() => 
            Layers.Select(layer => new double[layer.NeuronCount]).ToArray();

        private double[][,] CreateSynapses() =>
            Layers.Skip(1).Select((layer, index) => CreateSynapses(NeuronCount(index), layer.NeuronCount)).ToArray();

        private double[,] CreateSynapses(int inNeurons, int outNeurons)
        {
            var result = new double[inNeurons, outNeurons];
            Matrix.Map(result, (x, y) => random.NextDouble());
            return result;
        }

        private IReadOnlyList<NeuralNetworkLayer> Layers =>
            settings.Layers;

        private int LayerCount => 
            settings.Layers.Count;

        private double LearningRate =>
            settings.LearningRate;

        private int NeuronCount(int layerIndex) => 
            Layers[layerIndex].NeuronCount;

        private bool IsLastLayer(int layerIndex) =>
            layerIndex == LayerCount - 1;
    }
}