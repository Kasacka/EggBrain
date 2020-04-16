namespace EggBrain
{
    public sealed class NeuralNetworkLayerSettings
    {
        public NeuralNetworkLayerSettings(int neuronCount, IActivationFunction activationFunction)
        {
            NeuronCount = neuronCount;
            ActivationFunction = activationFunction;
        }

        public int NeuronCount { get; }
        public IActivationFunction ActivationFunction { get; }
    }
}