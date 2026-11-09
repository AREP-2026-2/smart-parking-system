package edu.eci.arep.parking.prediction;

import edu.eci.arep.parking.availability.OccupancyHistory;
import edu.eci.arep.parking.availability.OccupancyHistory.HourRow;
import edu.eci.arep.parking.prediction.PredictionDtos.CalendarStep;
import edu.eci.arep.parking.prediction.PredictionDtos.ForecastRequest;
import edu.eci.arep.parking.prediction.PredictionDtos.HistoryStep;
import edu.eci.arep.parking.prediction.PredictionDtos.Query;
import edu.eci.arep.parking.prediction.PredictionDtos.Result;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClientException;

/**
 * Arma las consultas con las series del Módulo 3 y garantiza una respuesta:
 * si el servicio de inferencia no contesta, responde la línea base local.
 */
@Service
public class ForecastGateway {

    public record ZoneHorizon(String zone, LocalDateTime asOf, double horizonHours) {
    }

    private static final Logger LOG = LoggerFactory.getLogger(ForecastGateway.class);
    private static final int WINDOW_HOURS = 24;
    private static final int WEEK_HOURS = 168;

    private final OccupancyHistory history;
    private final PredictionClient client;

    public ForecastGateway(OccupancyHistory history, PredictionClient client) {
        this.history = history;
        this.client = client;
    }

    public List<Result> forecast(List<ZoneHorizon> items, boolean includeAttention, boolean shadow) {
        List<Query> queries = items.stream().map(item -> query(item, shadow)).toList();
        try {
            return client.forecast(new ForecastRequest(queries, includeAttention)).results();
        } catch (RestClientException failure) {
            LOG.warn("Servicio de inferencia no disponible: {}", failure.getMessage());
            return items.stream().map(this::localBaseline).toList();
        }
    }

    private Query query(ZoneHorizon item, boolean shadow) {
        List<HistoryStep> steps = new ArrayList<>(WINDOW_HOURS);
        for (int back = WINDOW_HOURS - 1; back >= 0; back--) {
            HourRow row = history.at(item.zone(), item.asOf().minusHours(back));
            steps.add(new HistoryStep(row.time(), row.occupancy(), row.inflow() / row.capacity(),
                    row.outflow() / row.capacity(), row.classChange(), row.noClassDay(), row.rain()));
        }
        List<CalendarStep> calendar = new ArrayList<>(WINDOW_HOURS);
        for (int ahead = 1; ahead <= WINDOW_HOURS; ahead++) {
            HourRow row = history.at(item.zone(), item.asOf().plusHours(ahead));
            calendar.add(new CalendarStep(row.classChange(), row.noClassDay()));
        }
        return new Query(item.zone(), item.asOf(), item.horizonHours(), steps, calendar, weekAgo(item), shadow);
    }

    private double weekAgo(ZoneHorizon item) {
        LocalDateTime target = item.asOf().plusHours(Math.round(item.horizonHours()));
        return history.at(item.zone(), target.minusHours(WEEK_HOURS)).occupancy();
    }

    private Result localBaseline(ZoneHorizon item) {
        LocalDateTime target = item.asOf().plusSeconds(Math.round(item.horizonHours() * 3600));
        return new Result(item.zone(), item.horizonHours(), target, weekAgo(item), "baseline",
                "same-hour-last-week", true, "prediction_service_unavailable", null);
    }
}
