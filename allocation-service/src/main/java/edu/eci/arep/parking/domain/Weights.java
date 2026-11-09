package edu.eci.arep.parking.domain;

import jakarta.validation.constraints.PositiveOrZero;

/** Pesos de la Ecuación (2): α·D + β·T + γ·C + δ·I + ε·A + ζ·S. */
public record Weights(
        @PositiveOrZero double alpha,
        @PositiveOrZero double beta,
        @PositiveOrZero double gamma,
        @PositiveOrZero double delta,
        @PositiveOrZero double epsilon,
        @PositiveOrZero double zeta) {

    public Weights withoutSaturation() {
        return new Weights(alpha, beta, gamma, delta, epsilon, 0);
    }
}
