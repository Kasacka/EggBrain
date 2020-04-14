namespace neural
{
    public sealed class LinearActivationFunction : IActivationFunction
    {
        public double Perform(double value) => value;
        public double PerformDerivation(double _) => 1;
    }
}