package edu.eci.arep.parking.allocation;

import edu.eci.arep.parking.availability.OccupancyHistory;
import edu.eci.arep.parking.availability.OccupancyHistory.HourRow;
import edu.eci.arep.parking.availability.OccupancyHistory.ZoneInfo;
import edu.eci.arep.parking.availability.ParkingLayout;
import edu.eci.arep.parking.domain.CostFunction;
import edu.eci.arep.parking.domain.ScoredSpot;
import edu.eci.arep.parking.domain.Spot;
import edu.eci.arep.parking.domain.SpotSize;
import edu.eci.arep.parking.domain.Vehicle;
import edu.eci.arep.parking.domain.Weights;
import edu.eci.arep.parking.domain.ZoneContext;
import edu.eci.arep.parking.prediction.ForecastGateway;
import edu.eci.arep.parking.prediction.ForecastGateway.ZoneHorizon;
import edu.eci.arep.parking.prediction.PredictionDtos.Result;
import java.time.DayOfWeek;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import org.springframework.stereotype.Service;

/**
 * Hipótesis 3: simula el Escenario B sobre las horas pico del periodo de prueba.
 *
 * Todas las políticas atienden exactamente los mismos vehículos. Cada zona tiene una
 * vía de acceso con un tiempo de servicio fijo, de modo que concentrar vehículos en
 * una zona genera cola. Un vehículo "adecuado" estaciona con su tamaño y accesibilidad
 * correctos en una zona que no supera el 90% de ocupación al llegar.
 */
@Service
public class AllocationExperiment {

    public enum Policy { NEAREST, COST_WITHOUT_SATURATION, COST_REACTIVE, COST_FORECAST }

    public record PolicySummary(Policy policy, double adequatePercent, double meanSeconds) {
    }

    public record Difference(double mean, double lower95, double upper95) {
    }

    public record Result3(String datasetKind, int events, int vehicles, int degradedEvents,
                          List<PolicySummary> policies, Difference adequateVsNearest,
                          Difference secondsVsNearest, boolean hypothesisHolds) {
    }

    private record Arrival(double minute, Vehicle vehicle) {
    }

    private record Outcome(double adequatePercent, double meanSeconds) {
    }

    private static final int[] PEAK_HOURS = {6, 9, 12};
    private static final double SERVICE_SECONDS = 15;
    private static final double SATURATED = 0.90;
    private static final double NO_SPOT_SECONDS = 600;
    private static final double QUEUE_FOR_FULL_CONGESTION_SECONDS = 120;
    private static final double ACCESSIBLE_SHARE = 0.03;
    private static final long SEED = 42;

    private final OccupancyHistory history;
    private final ParkingLayout layout;
    private final ForecastGateway gateway;

    public AllocationExperiment(OccupancyHistory history, ParkingLayout layout, ForecastGateway gateway) {
        this.history = history;
        this.layout = layout;
        this.gateway = gateway;
    }

    public Result3 run(Weights weights) {
        List<LocalDateTime> events = peakEvents();
        Map<Policy, List<Outcome>> outcomes = new HashMap<>();
        for (Policy policy : Policy.values()) {
            outcomes.put(policy, new ArrayList<>());
        }
        int vehicles = 0;
        int degradedEvents = 0;
        for (LocalDateTime event : events) {
            Random random = new Random(SEED + event.hashCode());
            List<Arrival> arrivals = arrivals(event, random);
            vehicles += arrivals.size();
            List<ZoneHorizon> queries = history.zones().stream()
                    .map(zone -> new ZoneHorizon(zone.id(), event.minusHours(1), 1.0)).toList();
            List<Result> forecasts = gateway.forecast(queries, false, false);
            if (forecasts.stream().anyMatch(Result::degraded)) {
                degradedEvents++;
            }
            Map<String, Double> predicted = new HashMap<>();
            forecasts.forEach(forecast -> predicted.put(forecast.zone(), forecast.occupancy()));
            for (Policy policy : Policy.values()) {
                outcomes.get(policy).add(simulate(policy, event, arrivals, predicted, weights));
            }
        }
        return summarize(events.size(), vehicles, degradedEvents, outcomes);
    }

    private List<LocalDateTime> peakEvents() {
        List<LocalDateTime> events = new ArrayList<>();
        String zone = history.zones().get(0).id();
        for (LocalDateTime day = history.testStart().toLocalDate().atStartOfDay();
             !day.isAfter(history.lastQueryable()); day = day.plusDays(1)) {
            if (day.getDayOfWeek().compareTo(DayOfWeek.FRIDAY) > 0 || history.at(zone, day).noClassDay() > 0) {
                continue;
            }
            for (int hour : PEAK_HOURS) {
                LocalDateTime event = day.withHour(hour);
                if (!event.minusHours(1).isBefore(history.testStart()) && !event.isAfter(history.lastQueryable())) {
                    events.add(event);
                }
            }
        }
        return events;
    }

    /** Las llegadas reales de la hora, concentradas alrededor de la mitad de la hora. */
    private List<Arrival> arrivals(LocalDateTime event, Random random) {
        double total = history.zones().stream().mapToDouble(zone -> history.at(zone.id(), event).inflow()).sum();
        List<Arrival> arrivals = new ArrayList<>();
        for (int i = 0; i < (int) total; i++) {
            double minute = Math.max(0, Math.min(59.9, 30 + random.nextGaussian() * 12));
            double sizeDraw = random.nextDouble();
            SpotSize size = sizeDraw < 0.30 ? SpotSize.COMPACT : sizeDraw < 0.85 ? SpotSize.MEDIUM : SpotSize.LARGE;
            arrivals.add(new Arrival(minute, new Vehicle(size, random.nextDouble() < ACCESSIBLE_SHARE)));
        }
        arrivals.sort(Comparator.comparingDouble(Arrival::minute));
        return arrivals;
    }

    private Outcome simulate(Policy policy, LocalDateTime event, List<Arrival> arrivals,
                             Map<String, Double> predicted, Weights weights) {
        Map<String, Integer> counts = new HashMap<>();
        Map<String, Integer> capacity = new HashMap<>();
        Map<String, Double> laneFreeAt = new HashMap<>();
        Set<String> taken = new HashSet<>();
        for (ZoneInfo zone : history.zones()) {
            // Estado de partida: lo que había antes de la hora pico menos las salidas de esa hora.
            HourRow before = history.at(zone.id(), event.minusHours(1));
            int remaining = Math.max(0, before.occupied() - (int) history.at(zone.id(), event).outflow());
            counts.put(zone.id(), remaining);
            capacity.put(zone.id(), zone.capacity());
            laneFreeAt.put(zone.id(), 0.0);
            taken.addAll(layout.occupiedSpots(zone.id(), remaining, event));
        }
        List<Spot> free = new ArrayList<>(layout.spots().stream().filter(spot -> !taken.contains(spot.id())).toList());

        int adequate = 0;
        double totalSeconds = 0;
        for (Arrival arrival : arrivals) {
            double now = arrival.minute() * 60;
            Spot chosen = choose(policy, free, arrival.vehicle(), now, counts, capacity, laneFreeAt, predicted, weights);
            if (chosen == null) {
                totalSeconds += NO_SPOT_SECONDS;
                continue;
            }
            free.remove(chosen);
            int occupied = counts.merge(chosen.zone(), 1, Integer::sum);
            double reachesLane = now + CostFunction.travelMinutes(chosen.distanceMeters(), 0) * 60;
            double served = Math.max(reachesLane, laneFreeAt.get(chosen.zone())) + SERVICE_SECONDS;
            laneFreeAt.put(chosen.zone(), served);
            totalSeconds += served - now;
            boolean accessibilityOk = !arrival.vehicle().needsAccessible() || chosen.accessible();
            if (accessibilityOk && (double) occupied / capacity.get(chosen.zone()) <= SATURATED) {
                adequate++;
            }
        }
        int size = Math.max(1, arrivals.size());
        return new Outcome(100.0 * adequate / size, totalSeconds / size);
    }

    private Spot choose(Policy policy, List<Spot> free, Vehicle vehicle, double now, Map<String, Integer> counts,
                        Map<String, Integer> capacity, Map<String, Double> laneFreeAt,
                        Map<String, Double> predicted, Weights weights) {
        if (policy == Policy.NEAREST) {
            return CostFunction.nearest(free, vehicle).orElse(null);
        }
        Map<String, ZoneContext> contexts = new HashMap<>();
        for (Map.Entry<String, Integer> zone : counts.entrySet()) {
            double live = (double) zone.getValue() / capacity.get(zone.getKey());
            double queue = Math.max(0, laneFreeAt.get(zone.getKey()) - now);
            double congestion = Math.min(1, queue / QUEUE_FOR_FULL_CONGESTION_SECONDS);
            double saturation = switch (policy) {
                case COST_REACTIVE -> live;
                case COST_FORECAST -> Math.max(predicted.get(zone.getKey()), live);
                default -> 0;
            };
            contexts.put(zone.getKey(), new ZoneContext(congestion, saturation));
        }
        Weights used = policy == Policy.COST_WITHOUT_SATURATION ? weights.withoutSaturation() : weights;
        List<ScoredSpot> ranked = CostFunction.rank(free, vehicle, contexts, used);
        return ranked.isEmpty() ? null : ranked.get(0).spot();
    }

    private Result3 summarize(int events, int vehicles, int degradedEvents, Map<Policy, List<Outcome>> outcomes) {
        List<PolicySummary> summaries = new ArrayList<>();
        for (Policy policy : Policy.values()) {
            List<Outcome> list = outcomes.get(policy);
            summaries.add(new PolicySummary(policy,
                    list.stream().mapToDouble(Outcome::adequatePercent).average().orElse(Double.NaN),
                    list.stream().mapToDouble(Outcome::meanSeconds).average().orElse(Double.NaN)));
        }
        Difference adequate = pairedDifference(outcomes, true);
        Difference seconds = pairedDifference(outcomes, false);
        return new Result3(history.datasetKind(), events, vehicles, degradedEvents, summaries, adequate, seconds,
                adequate.lower95() > 0 && seconds.upper95() < 0);
    }

    /** Diferencia pareada por evento, forecast menos vecino más cercano, con IC del 95%. */
    private static Difference pairedDifference(Map<Policy, List<Outcome>> outcomes, boolean adequate) {
        List<Outcome> forecast = outcomes.get(Policy.COST_FORECAST);
        List<Outcome> nearest = outcomes.get(Policy.NEAREST);
        double[] differences = new double[forecast.size()];
        for (int i = 0; i < differences.length; i++) {
            differences[i] = adequate
                    ? forecast.get(i).adequatePercent() - nearest.get(i).adequatePercent()
                    : forecast.get(i).meanSeconds() - nearest.get(i).meanSeconds();
        }
        int n = differences.length;
        if (n < 2) {
            return new Difference(Double.NaN, Double.NaN, Double.NaN);
        }
        double mean = java.util.Arrays.stream(differences).average().orElse(Double.NaN);
        double variance = java.util.Arrays.stream(differences).map(d -> (d - mean) * (d - mean)).sum() / (n - 1);
        double margin = 1.96 * Math.sqrt(variance / n);
        return new Difference(mean, mean - margin, mean + margin);
    }
}
