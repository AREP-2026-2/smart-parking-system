package edu.eci.arep.parking.prediction;

import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;
import java.time.LocalDateTime;
import java.util.List;

/** Contrato JSON del servicio de inferencia. */
public final class PredictionDtos {

    private PredictionDtos() {
    }

    @JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
    public record HistoryStep(LocalDateTime timestamp, double occupancy, double inflow, double outflow,
                              double classChange, double noClassDay, double rain) {
    }

    @JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
    public record CalendarStep(double classChange, double noClassDay) {
    }

    @JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
    public record Query(String zone, LocalDateTime asOf, double horizonHours, List<HistoryStep> history,
                        List<CalendarStep> futureCalendar, double weekAgoOccupancy, boolean ignoreErrorLimit) {
    }

    @JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
    public record ForecastRequest(List<Query> queries, boolean includeAttention) {
    }

    @JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
    public record Result(String zone, double horizonHours, LocalDateTime targetTime, double occupancy,
                         String branch, String model, boolean degraded, String reason, List<Double> attention) {
    }

    public record ForecastResponse(List<Result> results) {
    }

    public record FeedbackItem(String branch, int horizon, double predicted, double observed) {
    }

    public record FeedbackRequest(List<FeedbackItem> items) {
    }
}
