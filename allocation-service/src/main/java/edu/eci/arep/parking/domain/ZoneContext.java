package edu.eci.arep.parking.domain;

/** Condiciones de una zona que entran en la función de costo. Valores en [0, 1]. */
public record ZoneContext(double congestion, double saturation) {
}
