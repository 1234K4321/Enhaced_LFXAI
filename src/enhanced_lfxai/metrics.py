import numpy as np
from scipy.integrate import quad


def faithfulness(predictor, explanation_function, x, subset_indices=0, reference_baseline=0):
    # Compute the attributions of the explanation function for the input x
    attributions = explanation_function(x)

    # Select the subset of features xs from the input x
    xs = x[subset_indices]

    # Set features in xs to a reference baseline while keeping other features unchanged
    modified_input = np.copy(x)
    modified_input[subset_indices] = reference_baseline

    # Compute the output of the predictor for the original and modified inputs
    f_x = predictor(x)
    f_modified = predictor(modified_input)

    # Compute the correlation between the sum of attributions and the difference in output
    attribution_sum = np.sum(attributions[subset_indices])
    output_diff = f_x - f_modified

    # Compute the correlation coefficient
    correlation = np.corrcoef(attribution_sum, output_diff)[0, 1]

    return correlation


def average_sensitivity(predictor, explanation_function, distance_metric, radius, x, distribution):
    def integrand(z):
        # Compute the distance between the explanation function outputs for x and z
        distance = distance_metric(explanation_function(predictor, x), explanation_function(predictor, z))

        # Evaluate the distribution at z
        p_z = distribution(z)

        return distance * p_z

    # Define the neighborhood Nr within radius r of x
    def neighborhood_condition(z):
        return distance_metric(x, z) <= radius and predictor(x) == predictor(z)

    # Integrate over the neighborhood Nr
    integral_value, _ = quad(integrand, -np.inf, np.inf, weight=neighborhood_condition)

    return integral_value


def average_pearson_correlation(feature_importance_scores):
    # Initialize list to store Pearson correlation coefficients
    correlations = []

    # Compute Pearson correlation coefficient between all pairs of feature importance scores
    for i in range(len(feature_importance_scores)):
        for j in range(i + 1, len(feature_importance_scores)):
            corr = np.corrcoef(feature_importance_scores[i], feature_importance_scores[j])[0, 1]
            correlations.append(corr)

    # Compute the average Pearson correlation coefficient
    average_correlation = np.mean(correlations)

    return average_correlation

