using System;
using System.Collections.Generic;
using System.Linq;

namespace EggBrain
{
    public sealed class NeuralNetworkSettings
    {
        internal const double MaximumLearningRate = 1;

        public NeuralNetworkSettings(double learningRate, IEnumerable<NeuralNetworkLayerSettings> layers)
        {
            Guard.NotNegative(learningRate, nameof(learningRate));
            Guard.Maximum(learningRate, MaximumLearningRate, nameof(learningRate));
            Guard.NotNull(layers, nameof(layers));
            if (layers.Count() == 0)
                throw new ArgumentException("There must be at least one layer");
            LearningRate = learningRate;
            Layers = layers.ToArray();
        }

        public double LearningRate { get; }
        public IReadOnlyList<NeuralNetworkLayerSettings> Layers { get; }
    }
}