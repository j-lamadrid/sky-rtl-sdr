// Generated from best_gmm
// Config: 5 Components, full Covariance
// Feature Order: [0] SNR, [1] Log Duration, [2] Log Rise Time

#define NUM_CLUSTERS 5
#define FEATURE_DIM 3

// Cluster 0: METEOR_OVERDENSE
// Cluster 1: Noise (Sharp)
// Cluster 2: METEOR_UNDERDENSE
// Cluster 3: METEOR_OVERDENSE
// Cluster 4: Noise (Sharp)

// Means [Clusters][Features]
const float MODEL_MEANS[5][3] = {
    { 16.811125, -0.843815, -1.609438 },
    { 13.600486, -1.557676, -20.723266 },
    { 14.606570, -1.277939, -2.302585 },
    { 20.471856, 0.550591, -0.021578 },
    { 13.013822, -2.302585, -20.723266 }
};

// Precision Matrices [Clusters][Features*Features] (Flattened 3x3)
const float MODEL_PRECS[5][9] = {
    { 0.152035, -0.890337, -0.000000, -0.890337, 23.100163, 0.000002, -0.000000, 0.000002, 1000000.000000 },
    { 3.493871, -0.300686, -0.000000, -0.300686, 54.643068, 0.000000, -0.000000, 0.000000, 1000000.000000 },
    { 0.702099, -2.304492, 0.000000, -2.304492, 19.529041, 0.000001, 0.000000, 0.000001, 1000000.000000 },
    { 0.046049, -0.484439, 0.306843, -0.484439, 15.098133, -12.192347, 0.306843, -12.192347, 10.439785 },
    { 10.347812, 0.000000, -0.000000, 0.000000, 1000000.000000, 0.000000, -0.000000, 0.000000, 1000000.000000 }
};

// Log Determinants [Clusters]
const float MODEL_LOG_DETS[5] = { -14.815898, -19.066870, -15.943818, 1.795027, -29.967796 };

// Log Priors [Clusters]
const float MODEL_LOG_PRIORS[5] = { -2.127446, -2.208976, -1.263362, -2.967575, -0.828015 };

// PHYISCS INDICES MAPPING (Use these in your logic)
#define CLUSTER_0_TYPE  //METEOR_OVERDENSE
#define CLUSTER_1_TYPE  //NOISE_SHARP
#define CLUSTER_2_TYPE  //METEOR_UNDERDENSE
#define CLUSTER_3_TYPE  //METEOR_OVERDENSE
#define CLUSTER_4_TYPE  //NOISE_SHARP
