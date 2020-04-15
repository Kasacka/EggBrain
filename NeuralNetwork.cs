using System;
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
            Layers = settings.Layers.ToArray();
            layerInputs = CreateLayers(Layers);
            layerOutputs = CreateLayers(Layers);
            deltas = CreateLayers(Layers);
            synapses = CreateSynapses(Layers);
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
            Activate(layerIndex);
            if (layerIndex < LayerCount - 1)
                layerInputs[layerIndex + 1] = Maths.DotProduct(synapses[layerIndex], layerOutputs[layerIndex]);
        }

        private void Activate(int layerIndex)
        {
            var function = Layers[layerIndex].ActivationFunction;
            Maths.Map(layerInputs[layerIndex], function.Perform, layerOutputs[layerIndex]);
        }

        private void PropagateBackward(int layerIndex, double[] errors)
        {
            var derivations = layerInputs[layerIndex].Select(Layers[layerIndex].ActivationFunction.PerformDerivation).ToArray();
            var differences = new double[Layers[layerIndex].NeuronCount];
            if (layerIndex == LayerCount - 1)
            {
                Maths.Difference(layerOutputs[layerIndex], errors, differences);
            }
            else
            {
                for (var x = 0; x < Layers[layerIndex].NeuronCount; ++x) {
                    differences[x] = deltas[layerIndex + 1]
                        .Select((h, y) => h * synapses[layerIndex][x, y])
                        .Sum();
                }
            }

            deltas[layerIndex] = new double[Layers[layerIndex].NeuronCount];
            Maths.Multiply(differences, derivations, deltas[layerIndex]);
            var deltaWeights = layerOutputs[layerIndex - 1]
                .Select(output => deltas[layerIndex].Select(d => d * output * (-1) * settings.LearningRate).ToArray())
                .ToArray();
            
            Matrix.Map(synapses[layerIndex - 1], (x, y, current) => current + deltaWeights[x][y]);
        }

        private static double[][] CreateLayers(NeuralNetworkLayer[] layers) => 
            layers.Select(layer => new double[layer.NeuronCount]).ToArray();

        private static double[][,] CreateSynapses(NeuralNetworkLayer[] layers) =>
            layers.Skip(1).Select((layer, index) => CreateSynapses(layers[index].NeuronCount, layer.NeuronCount)).ToArray();

        private static double[,] CreateSynapses(int inNeurons, int outNeurons)
        {
            var result = new double[inNeurons, outNeurons];
            Matrix.Map(result, (x, y) => random.NextDouble());
            return result;
        }

        private NeuralNetworkLayer[] Layers { get; }

        private int LayerCount => 
            Layers.Length;

        private int NeuronCount(int layerIndex) => 
            Layers[layerIndex].NeuronCount;
    }
}