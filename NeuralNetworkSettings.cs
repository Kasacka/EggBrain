using System.Collections.Generic;
using System.Linq;

namespace EggBrain
{
    public sealed class NeuralNetworkSettings
    {
        public NeuralNetworkSettings(double learningRate, IEnumerable<NeuralNetworkLayer> layers)
        {
            LearningRate = learningRate;
            Layers = layers.ToArray();
        }

        public double LearningRate { get; }
        public IEnumerable<NeuralNetworkLayer> Layers { get; }
    }
}