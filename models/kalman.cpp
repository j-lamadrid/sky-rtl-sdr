#include "kalman.hpp"

KalmanFilter::KalmanFilter(double Q, double R, double P0)
    : Q(Q), R(R), P(P0), x(0.0), K(0.0) {}

KalmanFilter::KalmanFilter() {}

void KalmanFilter::init(double x0) {
    x = x0;
}

double KalmanFilter::update(double measurement) {
    // 1. Prediction Step
    // For a 1D noise tracker, we assume the state is constant (A=1)
    // x_pred = x_old
    // P_pred = P_old + Q
    P = P + Q;

    // 2. Update Step
    // Kalman Gain: How much do we trust this new measurement?
    // K = P / (P + R)
    K = P / (P + R);

    // New Estimate: x = x + K * (measurement - x)
    x = x + K * (measurement - x);

    // New Uncertainty: P = (1 - K) * P
    P = (1.0 - K) * P;

    return x;
}
