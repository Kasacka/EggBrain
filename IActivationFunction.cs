namespace neural
{
    public interface IActivationFunction
    {
        double Perform(double value);
        double PerformDerivation(double value);
    }
}