package edu.eci.arep.parking.allocation;

import edu.eci.arep.parking.availability.DigitalTwin;
import edu.eci.arep.parking.availability.DigitalTwin.ZoneState;
import edu.eci.arep.parking.config.ParkingProperties;
import edu.eci.arep.parking.domain.CostFunction;
import edu.eci.arep.parking.domain.ScoredSpot;
import edu.eci.arep.parking.domain.Spot;
import edu.eci.arep.parking.domain.Vehicle;
import edu.eci.arep.parking.domain.Weights;
import edu.eci.arep.parking.domain.ZoneContext;
import edu.eci.arep.parking.prediction.ForecastGateway;
import edu.eci.arep.parking.prediction.ForecastGateway.ZoneHorizon;
import edu.eci.arep.parking.prediction.PredictionDtos.Result;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.springframework.stereotype.Service;

/** Motor de asignación (Módulo 5): combina estado actual, congestión y ocupación prevista. */
@Service
public class AllocationService {

    private static final int ALTERNATIVES = 5;
    private static final double MIN_HORIZON_HOURS = 1.0 / 60;
    private static final double MAX_HORIZON_HOURS = 24;

    public record Assignment(boolean assigned, ScoredSpot best, ScoredSpot nearest, List<ScoredSpot> alternatives,
                             List<Result> forecasts, boolean degraded, boolean accessibilityMet,
                             Weights weights, long latencyMs, String message) {
    }

    private final DigitalTwin twin;
    private final ForecastGateway gateway;
    private final Weights defaultWeights;

    public AllocationService(DigitalTwin twin, ForecastGateway gateway, ParkingProperties properties) {
        this.twin = twin;
        this.gateway = gateway;
        this.defaultWeights = properties.weights();
    }

    public Weights defaultWeights() {
        return defaultWeights;
    }

    public Assignment assign(Vehicle vehicle, double arrivalInHours, Weights requested, boolean reserve) {
        long started = System.nanoTime();
        Weights weights = requested != null ? requested : defaultWeights;
        List<Spot> free = twin.freeSpots();
        List<Spot> candidates = CostFunction.candidates(free, vehicle);
        if (candidates.isEmpty()) {
            return new Assignment(false, null, null, List.of(), List.of(), false, false, weights,
                    elapsedMs(started), "No hay espacios compatibles con este vehículo");
        }

        Map<String, ZoneState> zones = twin.zones().stream()
                .collect(Collectors.toMap(ZoneState::id, Function.identity()));
        LocalDateTime asOf = twin.asOf();
        List<ZoneHorizon> requests = zoneHorizons(candidates, zones, asOf, arrivalInHours);
        List<Result> forecasts = gateway.forecast(requests, false, false);

        Map<String, ZoneContext> contexts = new HashMap<>();
        for (Result forecast : forecasts) {
            ZoneState zone = zones.get(forecast.zone());
            contexts.put(forecast.zone(), new ZoneContext(zone.congestion(), saturation(forecast, zone)));
        }

        List<ScoredSpot> ranked = CostFunction.rank(free, vehicle, contexts, weights);
        ScoredSpot best = ranked.get(0);
        Optional<Spot> nearestSpot = CostFunction.nearest(free, vehicle);
        ScoredSpot nearest = nearestSpot.map(spot ->
                CostFunction.score(spot, vehicle, contexts.get(spot.zone()), weights)).orElse(null);
        if (reserve) {
            twin.reserve(best.spot().id());
        }

        boolean accessibilityMet = !vehicle.needsAccessible() || best.spot().accessible();
        boolean degraded = forecasts.stream().anyMatch(Result::degraded);
        String message = accessibilityMet ? null
                : "No quedan espacios accesibles: se asigna el mejor espacio disponible";
        return new Assignment(true, best, nearest, ranked.subList(0, Math.min(ALTERNATIVES, ranked.size())),
                forecasts, degraded, accessibilityMet, weights, elapsedMs(started), message);
    }

    /** Un pronóstico por zona, en el instante en que el vehículo llegaría a su espacio más cercano. */
    private static List<ZoneHorizon> zoneHorizons(List<Spot> candidates, Map<String, ZoneState> zones,
                                                  LocalDateTime asOf, double arrivalInHours) {
        Map<String, Double> nearestByZone = new HashMap<>();
        for (Spot spot : candidates) {
            nearestByZone.merge(spot.zone(), spot.distanceMeters(), Math::min);
        }
        List<ZoneHorizon> requests = new ArrayList<>();
        nearestByZone.forEach((zone, distance) -> {
            double travelHours = CostFunction.travelMinutes(distance, zones.get(zone).congestion()) / 60;
            double horizon = Math.max(MIN_HORIZON_HOURS, Math.min(MAX_HORIZON_HOURS, arrivalInHours + travelHours));
            requests.add(new ZoneHorizon(zone, asOf, horizon));
        });
        return requests;
    }

    /** En el corto plazo, la saturación prevista no puede ser menor que la ya observada. */
    private static double saturation(Result forecast, ZoneState zone) {
        return forecast.horizonHours() <= 1 ? Math.max(forecast.occupancy(), zone.occupancy()) : forecast.occupancy();
    }

    private static long elapsedMs(long started) {
        return (System.nanoTime() - started) / 1_000_000;
    }
}
