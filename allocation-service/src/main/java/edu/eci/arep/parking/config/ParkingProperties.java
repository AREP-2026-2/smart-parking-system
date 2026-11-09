package edu.eci.arep.parking.config;

import edu.eci.arep.parking.domain.Weights;
import java.nio.file.Path;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties("parking")
public record ParkingProperties(Path dataDir, String predictionUrl, Weights weights) {
}
