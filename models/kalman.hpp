class KalmanFilter {
public:
    /**
     * @param Q  Process Noise (How much the system jitters naturally)
     * @param R  Measurement Noise (How much noise is in the sensor reading)
     * @param P0 Initial Estimation Uncertainty
     */
    KalmanFilter(double Q, double R, double P0);
    KalmanFilter();

    // Initialize the filter with a starting value
    void init(double x0);

    // Update the filter with a new measurement
    // Returns the new smoothed estimate
    double update(double measurement);

    // Getters
    double state() const { return x; }
    double uncertainty() const { return P; }

private:
    // Model Parameters
    double Q; // Process Variance
    double R; // Measurement Variance

    // System State
    double x; // Current Estimate (The Noise Floor)
    double P; // Current Error Covariance (Uncertainty)
    double K; // Kalman Gain
};
