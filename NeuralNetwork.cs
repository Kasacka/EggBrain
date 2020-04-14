using System;
using System.Linq;

namespace EggBrain
{
    public sealed class NeuralNetwork
    {
        private readonly NeuralNetworkSettings settings;
        private static readonly Random random;
        private readonly double[][] layerInputs;
        private readonly double[][] layerOutputs;
        private double[][,] synapses;
        private double[][] deltas;

        static NeuralNetwork() =>
            random = new Random();

        public NeuralNetwork(NeuralNetworkSettings settings) 
        {
            this.settings = settings;
            Layers = settings.Layers.ToArray();
            layerInputs = new double[LayerCount][];
            layerOutputs = new double[LayerCount][];
            InitializeLayers();
            InitializeSynapses();
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
            return layerOutputs[LayerCount - 1].ToList().ToArray();
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
            for (var neuronIndex = 0; neuronIndex < NeuronCount(layerIndex); ++neuronIndex)
                layerOutputs[layerIndex][neuronIndex] = function.Perform(layerInputs[layerIndex][neuronIndex]);
        }

        private void PropagateBackward(int layerIndex, double[] errors)
        {
            var layer = Layers[layerIndex];
            var inputs = layerInputs[layerIndex];
            var outputs = layerOutputs[layerIndex];
            var derivations = inputs.Select(layer.ActivationFunction.PerformDerivation).ToArray();
            var differences = new double[layer.NeuronCount];
            if (layerIndex == LayerCount - 1)
            {
                Maths.Difference(outputs, errors, differences);
            }
            else
            {
                for (var x = 0; x < layer.NeuronCount; ++x) {
                    differences[x] = deltas[layerIndex + 1]
                        .Select((h, y) => h * synapses[layerIndex][x, y])
                        .Sum();
                }
            }

            deltas[layerIndex] = new double[layer.NeuronCount];
            Maths.Multiply(differences, derivations, deltas[layerIndex]);
            var deltaWeights = layerOutputs[layerIndex - 1]
                .Select(output => deltas[layerIndex].Select(d => d * output * (-1) * settings.LearningRate).ToArray())
                .ToArray();

            for (var x = 0; x < deltaWeights.Length; ++x)
                for (var y = 0; y < deltaWeights[x].Length; ++y)
                    synapses[layerIndex - 1][x, y] += deltaWeights[x][y];
        }

        private void InitializeLayers()
        {
            for (var layerIndex = 0; layerIndex < LayerCount; ++layerIndex)
            {
                var neuronCount = Layers[layerIndex].NeuronCount;
                layerInputs[layerIndex] = new double[neuronCount];
                layerOutputs[layerIndex] = new double[neuronCount];
            }
        }

        private void InitializeSynapses()
        {
            synapses = new double[LayerCount - 1][,];
            deltas = new double[LayerCount][];
            for (var layerIndex = 0; layerIndex < synapses.Length; ++layerIndex)
            {
                var inNeurons = NeuronCount(layerIndex);
                var outNeurons = NeuronCount(layerIndex + 1);
                synapses[layerIndex] = new double[inNeurons, outNeurons];
                deltas[layerIndex] = new double[outNeurons];
                InitializeRandom(synapses[layerIndex]);
            }
        }

        private void InitializeRandom(double[,] matrix)
        {
            for (var x = 0; x < Maths.Width(matrix); ++x)
                for (var y = 0; y < Maths.Height(matrix); ++y)
                    matrix[x, y] = random.NextDouble();
        }

        private NeuralNetworkLayer[] Layers { get; }
        private int LayerCount => Layers.Length;
        private int NeuronCount(int layerIndex) => Layers[layerIndex].NeuronCount;
    }
}