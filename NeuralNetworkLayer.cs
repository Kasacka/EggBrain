namespace neural
{
    public sealed class NeuralNetworkLayer
    {
        public NeuralNetworkLayer(int neuronCount, IActivationFunction activationFunction)
        {
            NeuronCount = neuronCount;
            ActivationFunction = activationFunction;
        }

        public int NeuronCount { get; }
        public IActivationFunction ActivationFunction { get; }
    }
}