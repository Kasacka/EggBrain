using System;

namespace neural
{
    public sealed class SigmoidActivationFunction : IActivationFunction
    {
        public double Perform(double value) => 
            1d / (1d + Math.Exp(-value));

        public double PerformDerivation(double value)
        {
            var sigmoid = Perform(value);
            return sigmoid * (1 - sigmoid);
        }
    }
}