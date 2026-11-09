package edu.eci.arep.parking.observability;

import edu.eci.arep.parking.availability.DigitalTwin;
import edu.eci.arep.parking.availability.OccupancyHistory;
import edu.eci.arep.parking.availability.OccupancyHistory.ZoneInfo;
import edu.eci.arep.parking.prediction.ForecastGateway;
import edu.eci.arep.parking.prediction.ForecastGateway.ZoneHorizon;
import edu.eci.arep.parking.prediction.PredictionClient;
import edu.eci.arep.parking.prediction.PredictionDtos.FeedbackItem;
import edu.eci.arep.parking.prediction.PredictionDtos.FeedbackRequest;
import edu.eci.arep.parking.prediction.PredictionDtos.Result;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClientException;

/** Compara lo previsto con lo observado en las últimas horas y lo reporta al enrutador. */
@Service
public class ObservabilityService {

    public record HorizonError(int horizon, String branch, double mae, int samples) {
    }

    private static final int[] HORIZONS = {1, 3, 5, 8, 12, 24};
    private static final int TARGET_HOURS = 24;

    private final DigitalTwin twin;
    private final OccupancyHistory history;
    private final ForecastGateway gateway;
    private final PredictionClient client;

    public ObservabilityService(DigitalTwin twin, OccupancyHistory history, ForecastGateway gateway,
                                PredictionClient client) {
        this.twin = twin;
        this.history = history;
        this.gateway = gateway;
        this.client = client;
    }

    public List<HorizonError> backtest() {
        LocalDateTime asOf = twin.asOf();
        List<ZoneHorizon> items = new ArrayList<>();
        for (int horizon : HORIZONS) {
            for (int back = 0; back < TARGET_HOURS; back++) {
                LocalDateTime origin = asOf.minusHours(back).minusHours(horizon);
                for (ZoneInfo zone : history.zones()) {
                    items.add(new ZoneHorizon(zone.id(), origin, horizon));
                }
            }
        }
        List<Result> results = gateway.forecast(items, false, true);

        List<FeedbackItem> feedback = new ArrayList<>();
        Map<Integer, double[]> totals = new LinkedHashMap<>();
        Map<Integer, String> branches = new LinkedHashMap<>();
        for (int i = 0; i < results.size(); i++) {
            Result result = results.get(i);
            ZoneHorizon item = items.get(i);
            int horizon = (int) item.horizonHours();
            double observed = history.at(item.zone(), item.asOf().plusHours(horizon)).occupancy();
            double[] total = totals.computeIfAbsent(horizon, key -> new double[2]);
            total[0] += Math.abs(result.occupancy() - observed);
            total[1]++;
            branches.put(horizon, result.branch());
            if (!"baseline".equals(result.branch())) {
                feedback.add(new FeedbackItem(result.branch(), horizon, result.occupancy(), observed));
            }
        }
        try {
            if (!feedback.isEmpty()) {
                client.feedback(new FeedbackRequest(feedback));
            }
        } catch (RestClientException ignored) {
            // Sin servicio de inferencia no hay a quién reportarle; la tabla sigue siendo válida.
        }
        return totals.entrySet().stream().map(entry -> new HorizonError(entry.getKey(), branches.get(entry.getKey()),
                entry.getValue()[0] / entry.getValue()[1], (int) entry.getValue()[1])).toList();
    }
}
