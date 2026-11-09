package edu.eci.arep.parking.prediction;

import edu.eci.arep.parking.config.ParkingProperties;
import edu.eci.arep.parking.prediction.PredictionDtos.FeedbackRequest;
import edu.eci.arep.parking.prediction.PredictionDtos.ForecastRequest;
import edu.eci.arep.parking.prediction.PredictionDtos.ForecastResponse;
import java.time.Duration;
import org.springframework.boot.http.client.ClientHttpRequestFactoryBuilder;
import org.springframework.boot.http.client.ClientHttpRequestFactorySettings;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

@Component
public class HttpPredictionClient implements PredictionClient {

    private static final Duration CONNECT_TIMEOUT = Duration.ofMillis(500);
    private static final Duration READ_TIMEOUT = Duration.ofSeconds(5);

    private final RestClient client;

    public HttpPredictionClient(RestClient.Builder builder, ParkingProperties properties) {
        this.client = builder.baseUrl(properties.predictionUrl())
                // HTTP/1.1 explícito: uvicorn descarta el cuerpo si el cliente intenta negociar h2c.
                .requestFactory(ClientHttpRequestFactoryBuilder.simple().build(ClientHttpRequestFactorySettings.defaults()
                        .withConnectTimeout(CONNECT_TIMEOUT).withReadTimeout(READ_TIMEOUT)))
                .build();
    }

    @Override
    public ForecastResponse forecast(ForecastRequest request) {
        return client.post().uri("/v1/forecasts").contentType(MediaType.APPLICATION_JSON)
                .body(request).retrieve().body(ForecastResponse.class);
    }

    @Override
    public void feedback(FeedbackRequest request) {
        client.post().uri("/v1/feedback").contentType(MediaType.APPLICATION_JSON)
                .body(request).retrieve().toBodilessEntity();
    }
}
