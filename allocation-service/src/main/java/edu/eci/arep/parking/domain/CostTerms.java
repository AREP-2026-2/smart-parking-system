package edu.eci.arep.parking.domain;

/** Términos normalizados en [0, 1] de la función de costo de un espacio candidato. */
public record CostTerms(double distance, double time, double congestion,
                        double inadequacy, double accessibility, double saturation) {

    public double total(Weights w) {
        return w.alpha() * distance + w.beta() * time + w.gamma() * congestion
                + w.delta() * inadequacy + w.epsilon() * accessibility + w.zeta() * saturation;
    }
}
