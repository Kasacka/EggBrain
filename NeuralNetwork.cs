using System;
using System.Collections.Generic;
using System.Linq;

namespace neural
{
    public sealed class NeuralNetwork
    {
        private const double LearningRate = 0.1;
        private readonly NeuralNetworkLayer[] layers;
        private readonly Random random;
        private Matrix[] synapses;
        private double[][] deltas;
        private double[][] layerInputs;
        private double[][] layerOutputs;

        public NeuralNetwork(IEnumerable<NeuralNetworkLayer> layers) 
        {
            random = new Random();
            this.layers = layers.ToArray();
            InitializeLayers();
            InitializeSynapses();
        }

        public void Train(double[] inputs, double[] expectedOutputs)
        {
            //Console.WriteLine("INPUTS: " + string.Join("   ", inputs));
            Array.Copy(inputs, 0, layerInputs[0], 0, inputs.Length);
            //Console.WriteLine("INPUTS: " + string.Join("   ", layerInputs[0]));
            for (var layerIndex = 0; layerIndex < Layers; ++layerIndex)
                PropagateForward(layerIndex);
            //double[][] previousWeights = null;
            for (var layerIndex = Layers - 1; layerIndex >= 0; --layerIndex)
                PropagateBackward(layerIndex, expectedOutputs);
        }

        public double[] Test(double[] inputs)
        {
            Array.Copy(inputs, 0, layerInputs[0], 0, inputs.Length);
            for (var layerIndex = 0; layerIndex < Layers; ++layerIndex)
                PropagateForward(layerIndex);
            return layerOutputs[Layers - 1].ToList().ToArray();
        }

        private void PropagateForward(int layerIndex)
        {            
            //Console.WriteLine("Layer Index: " + layerIndex);
            var function = layers[layerIndex].ActivationFunction;
            //Console.WriteLine(function);
            layerOutputs[layerIndex] = layerInputs[layerIndex].Select(function.Perform).ToArray();
            //Console.WriteLine(function);
            //Console.WriteLine("Inputs: " + string.Join(",", layerInputs[layerIndex]));
            //Console.WriteLine("Inputs: " + string.Join(",", layerOutputs[layerIndex]));
            //Environment.Exit(0);
            if (layerIndex < Layers - 1)
                layerInputs[layerIndex + 1] = synapses[layerIndex] * layerOutputs[layerIndex];
        }

        private void PropagateBackward(int layerIndex, double[] errors /*, double[][] previousWeights*/)
        {
            if (layerIndex == 0)
                return;
            var layer = layers[layerIndex];
            var inputs = layerInputs[layerIndex];
            var outputs = layerOutputs[layerIndex];
            var derivations = inputs.Select(layer.ActivationFunction.PerformDerivation).ToArray();
            double[] differences = null;
            if (layerIndex == Layers - 1)
            {
                differences = Maths.Difference(outputs, errors);
                //Console.WriteLine(string.Join("  ", differences));
                //Environment.Exit(0);
            }
            else
            {
                //Console.WriteLine(4);
                var ll = new List<double>();
                for (var x = 0; x < layers[layerIndex].NeuronCount; ++x) {
                    var s = 0d;
                    for (var y = 0; y < layers[layerIndex + 1].NeuronCount; ++y) {
                        s += deltas[layerIndex + 1][y] * synapses[layerIndex/* + 1*/][x,y];
                    }
                    ll.Add(s);
                }
                differences = ll.ToArray();
            }

            deltas[layerIndex] = Maths.Multiply(differences, derivations);
            var deltaWeights = layerOutputs[layerIndex - 1]
                .Select(output => deltas[layerIndex].Select(d => d * output * (-1) * LearningRate).ToArray())
                .ToArray();

            for (var x = 0; x < deltaWeights.Length; ++x)
                for (var y = 0; y < deltaWeights[x].Length; ++y)
                    synapses[layerIndex - 1][x, y] += deltaWeights[x][y];
            

            //return weights;
        }

        private void InitializeLayers()
        {
            layerInputs = new double[Layers][];
            layerOutputs = new double[Layers][];
            for (var layerIndex = 0; layerIndex < Layers; ++layerIndex)
            {
                var neuronCount = layers[layerIndex].NeuronCount;
                layerInputs[layerIndex] = new double[neuronCount];
                layerOutputs[layerIndex] = new double[neuronCount];
            }
        }

        private void InitializeSynapses()
        {
            synapses = new Matrix[layers.Length - 1];
            deltas = new double[Layers][];
            for (var layerIndex = 0; layerIndex < synapses.Length; ++layerIndex)
            {
                var inLayer = this.layers[layerIndex];
                var outLayer = this.layers[layerIndex + 1];
                synapses[layerIndex] = new Matrix(inLayer.NeuronCount, outLayer.NeuronCount);
                deltas[layerIndex] = new double[outLayer.NeuronCount];
                InitializeRandom(synapses[layerIndex]);
            }
        }

        private void InitializeRandom(Matrix matrix)
        {
            for (var x = 0; x < matrix.Width; ++x)
                for (var y = 0; y < matrix.Height; ++y)
                    matrix[x, y] = random.NextDouble();
        }

        private int Layers =>
            layers.Length;
    }
}