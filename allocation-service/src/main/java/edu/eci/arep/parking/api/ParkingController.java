package edu.eci.arep.parking.api;

import edu.eci.arep.parking.allocation.AllocationExperiment;
import edu.eci.arep.parking.allocation.AllocationService;
import edu.eci.arep.parking.allocation.AllocationService.Assignment;
import edu.eci.arep.parking.api.ScenarioService.Scenario;
import edu.eci.arep.parking.availability.DigitalTwin;
import edu.eci.arep.parking.availability.DigitalTwin.SpotState;
import edu.eci.arep.parking.availability.DigitalTwin.ZoneState;
import edu.eci.arep.parking.availability.OccupancyHistory;
import edu.eci.arep.parking.domain.SpotSize;
import edu.eci.arep.parking.domain.Vehicle;
import edu.eci.arep.parking.domain.Weights;
import edu.eci.arep.parking.observability.ObservabilityService;
import edu.eci.arep.parking.observability.ObservabilityService.HorizonError;
import edu.eci.arep.parking.prediction.ForecastGateway;
import edu.eci.arep.parking.prediction.ForecastGateway.ZoneHorizon;
import edu.eci.arep.parking.prediction.PredictionDtos.Result;
import jakarta.validation.Valid;
import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.NotNull;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
public class ParkingController {

    private static final int[] HORIZONS = {1, 3, 5, 8, 12, 24};

    public record State(String datasetKind, LocalDateTime asOf, LocalDateTime minAsOf, LocalDateTime maxAsOf,
                        List<ZoneState> zones, Weights defaultWeights, List<SpotState> spots) {
    }

    public record ClockRequest(@NotNull LocalDateTime asOf) {
    }

    public record AssignmentRequest(@NotNull SpotSize size, boolean needsAccessible,
                                    @DecimalMin("0") @DecimalMax("23.5") double arrivalInHours,
                                    @Valid Weights weights, boolean reserve) {
    }

    public record HistoryPoint(LocalDateTime time, double occupancy) {
    }

    public record ForecastPoint(Result forecast, double observed) {
    }

    public record ForecastView(String zone, List<HistoryPoint> history, List<HistoryPoint> future,
                               List<ForecastPoint> points) {
    }

    private final DigitalTwin twin;
    private final OccupancyHistory history;
    private final AllocationService allocation;
    private final AllocationExperiment experiment;
    private final ForecastGateway gateway;
    private final ScenarioService scenarios;
    private final ObservabilityService observability;

    public ParkingController(DigitalTwin twin, OccupancyHistory history, AllocationService allocation,
                             AllocationExperiment experiment, ForecastGateway gateway,
                             ScenarioService scenarios, ObservabilityService observability) {
        this.twin = twin;
        this.history = history;
        this.allocation = allocation;
        this.experiment = experiment;
        this.gateway = gateway;
        this.scenarios = scenarios;
        this.observability = observability;
    }

    @GetMapping("/state")
    public State state() {
        return new State(history.datasetKind(), twin.asOf(), history.testStart(), history.lastQueryable(),
                twin.zones(), allocation.defaultWeights(), twin.spots());
    }

    @PutMapping("/clock")
    public State moveClock(@Valid @RequestBody ClockRequest request) {
        twin.moveTo(request.asOf());
        return state();
    }

    @PostMapping("/reset")
    public State reset() {
        twin.reset();
        return state();
    }

    @PostMapping("/scenarios/{id}")
    public Scenario scenario(@PathVariable String id) {
        return scenarios.load(id);
    }

    @PostMapping("/assignments")
    public Assignment assign(@Valid @RequestBody AssignmentRequest request) {
        return allocation.assign(new Vehicle(request.size(), request.needsAccessible()),
                request.arrivalInHours(), request.weights(), request.reserve());
    }

    @GetMapping("/forecasts")
    public ForecastView forecasts(@RequestParam String zone) {
        boolean known = history.zones().stream().anyMatch(info -> info.id().equals(zone));
        if (!known) {
            throw new IllegalArgumentException("Zona desconocida: " + zone);
        }
        LocalDateTime asOf = twin.asOf();
        List<HistoryPoint> past = new ArrayList<>();
        for (int back = 23; back >= 0; back--) {
            LocalDateTime time = asOf.minusHours(back);
            past.add(new HistoryPoint(time, history.at(zone, time).occupancy()));
        }
        List<ZoneHorizon> queries = new ArrayList<>();
        for (int horizon : HORIZONS) {
            queries.add(new ZoneHorizon(zone, asOf, horizon));
        }
        List<ForecastPoint> points = gateway.forecast(queries, true, false).stream()
                .map(result -> new ForecastPoint(result,
                        history.at(zone, asOf.plusHours((long) result.horizonHours())).occupancy()))
                .toList();
        List<HistoryPoint> future = new ArrayList<>();
        for (int ahead = 1; ahead <= 24; ahead++) {
            LocalDateTime time = asOf.plusHours(ahead);
            future.add(new HistoryPoint(time, history.at(zone, time).occupancy()));
        }
        return new ForecastView(zone, past, future, points);
    }

    @PostMapping("/experiments/allocation")
    public AllocationExperiment.Result3 runExperiment(@Valid @RequestBody(required = false) Weights weights) {
        return experiment.run(weights != null ? weights : allocation.defaultWeights());
    }

    @PostMapping("/observability/backtest")
    public List<HorizonError> backtest() {
        return observability.backtest();
    }

    @ExceptionHandler(IllegalArgumentException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public Map<String, String> badRequest(IllegalArgumentException failure) {
        return Map.of("error", failure.getMessage());
    }
}
