namespace EggBrain
{
    public sealed class NeuralNetworkLayerSettings
    {
        internal const int MaximumNeuronCount = 500;

        public NeuralNetworkLayerSettings(int neuronCount, IActivationFunction activationFunction)
        {
            Guard.NotNegative(neuronCount, nameof(neuronCount));
            Guard.Maximum(neuronCount, MaximumNeuronCount, nameof(neuronCount));
            Guard.NotNull(activationFunction, nameof(activationFunction));
            NeuronCount = neuronCount;
            ActivationFunction = activationFunction;
        }

        public int NeuronCount { get; }
        public IActivationFunction ActivationFunction { get; }
    }
}