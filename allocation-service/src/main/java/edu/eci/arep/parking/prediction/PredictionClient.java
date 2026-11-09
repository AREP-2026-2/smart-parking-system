package edu.eci.arep.parking.prediction;

import edu.eci.arep.parking.prediction.PredictionDtos.FeedbackRequest;
import edu.eci.arep.parking.prediction.PredictionDtos.ForecastRequest;
import edu.eci.arep.parking.prediction.PredictionDtos.ForecastResponse;

/** Puerto hacia el motor predictivo: la asignación no sabe qué modelo responde. */
public interface PredictionClient {

    ForecastResponse forecast(ForecastRequest request);

    void feedback(FeedbackRequest request);
}
