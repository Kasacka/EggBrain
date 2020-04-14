namespace EggBrain
{
    public interface IActivationFunction
    {
        double Perform(double value);
        double PerformDerivation(double value);
    }
}