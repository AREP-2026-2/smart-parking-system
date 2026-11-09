package edu.eci.arep.parking;

import edu.eci.arep.parking.config.ParkingProperties;
import edu.eci.arep.parking.domain.Weights;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/** Crea un directorio de datos mínimo: dos zonas, tres semanas, ocupación constante. */
public final class TestData {

    public static final LocalDateTime START = LocalDateTime.of(2026, 5, 4, 0, 0);
    public static final LocalDateTime TEST_START = START.plusDays(14);
    public static final Weights WEIGHTS = new Weights(0.25, 0.20, 0.15, 0.10, 0.10, 0.20);
    public static final int ZONE_A_CAPACITY = 40;
    public static final int ZONE_B_CAPACITY = 40;

    private static final DateTimeFormatter FORMAT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    private TestData() {
    }

    public static ParkingProperties create(Path directory, int occupiedA, int occupiedB) throws IOException {
        Files.writeString(directory.resolve("zones.json"),
                "[{\"id\":\"A\",\"name\":\"Zona A\",\"capacity\":40},{\"id\":\"B\",\"name\":\"Zona B\",\"capacity\":40}]");
        Files.writeString(directory.resolve("metadata.json"),
                "{\"dataset_kind\":\"synthetic\",\"test_start\":\"%s\"}".formatted(TEST_START.format(FORMAT)));
        StringBuilder csv = new StringBuilder("zone,timestamp,occupied,capacity,inflow,outflow,class_change,no_class_day,rain\n");
        for (String zone : new String[] {"A", "B"}) {
            int occupied = "A".equals(zone) ? occupiedA : occupiedB;
            for (int hour = 0; hour < 24 * 21; hour++) {
                LocalDateTime time = START.plusHours(hour);
                csv.append("%s,%s,%d,40,4,1,0.0,0,0%n".formatted(zone, time.format(FORMAT), occupied));
            }
        }
        Files.writeString(directory.resolve("occupancy_hourly.csv"), csv.toString());
        return new ParkingProperties(directory, "http://unused", WEIGHTS);
    }
}
