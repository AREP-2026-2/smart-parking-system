package edu.eci.arep.parking.domain;

import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/** Función de costo multicriterio de la Ecuación (2). Sin estado y sin dependencias. */
public final class CostFunction {

    public static final double MAX_DISTANCE_METERS = 700;
    public static final double MAX_TRAVEL_MINUTES = 15;
    private static final double SPEED_METERS_PER_SECOND = 2.0;
    private static final double CONGESTION_SLOWDOWN = 2.0;

    private CostFunction() {
    }

    public static double travelMinutes(double distanceMeters, double congestion) {
        return distanceMeters / SPEED_METERS_PER_SECOND / 60 * (1 + CONGESTION_SLOWDOWN * congestion);
    }

    /** Filtro duro: el vehículo debe caber y un espacio accesible solo es para quien lo necesita. */
    public static boolean eligible(Spot spot, Vehicle vehicle) {
        return spot.size().fits(vehicle.size()) && (vehicle.needsAccessible() || !spot.accessible());
    }

    /** Candidatos válidos. Quien necesita accesibilidad solo recibe otro si no queda ninguno accesible. */
    public static List<Spot> candidates(List<Spot> free, Vehicle vehicle) {
        List<Spot> eligible = free.stream().filter(spot -> eligible(spot, vehicle)).toList();
        if (vehicle.needsAccessible() && eligible.stream().anyMatch(Spot::accessible)) {
            return eligible.stream().filter(Spot::accessible).toList();
        }
        return eligible;
    }

    public static ScoredSpot score(Spot spot, Vehicle vehicle, ZoneContext zone, Weights weights) {
        double minutes = travelMinutes(spot.distanceMeters(), zone.congestion());
        CostTerms terms = new CostTerms(
                clamp(spot.distanceMeters() / MAX_DISTANCE_METERS),
                clamp(minutes / MAX_TRAVEL_MINUTES),
                clamp(zone.congestion()),
                (spot.size().ordinal() - vehicle.size().ordinal()) / 2.0,
                vehicle.needsAccessible() && !spot.accessible() ? 1 : 0,
                clamp(zone.saturation()));
        return new ScoredSpot(spot, terms, terms.total(weights), minutes);
    }

    /** Candidatos de menor a mayor costo; los empates se resuelven por identificador. */
    public static List<ScoredSpot> rank(List<Spot> free, Vehicle vehicle,
                                        Map<String, ZoneContext> zones, Weights weights) {
        return candidates(free, vehicle).stream()
                .map(spot -> score(spot, vehicle, zones.get(spot.zone()), weights))
                .sorted(Comparator.comparingDouble(ScoredSpot::cost)
                        .thenComparing(scored -> scored.spot().id()))
                .toList();
    }

    /** Heurística de referencia: el candidato válido geométricamente más cercano. */
    public static Optional<Spot> nearest(List<Spot> free, Vehicle vehicle) {
        return candidates(free, vehicle).stream()
                .min(Comparator.comparingDouble(Spot::distanceMeters).thenComparing(Spot::id));
    }

    private static double clamp(double value) {
        return Math.max(0, Math.min(1, value));
    }
}
