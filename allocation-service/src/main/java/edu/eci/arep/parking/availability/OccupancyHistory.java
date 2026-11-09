package edu.eci.arep.parking.availability;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import edu.eci.arep.parking.config.ParkingProperties;
import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.springframework.stereotype.Component;

/** Histórico horario por zona del Módulo 3: la serie que consume el motor predictivo. */
@Component
public class OccupancyHistory {

    private static final DateTimeFormatter CSV_TIME = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    public record ZoneInfo(String id, String name, int capacity) {
    }

    public record HourRow(LocalDateTime time, int occupied, int capacity, double inflow, double outflow,
                          double classChange, double noClassDay, double rain) {
        public double occupancy() {
            return (double) occupied / capacity;
        }
    }

    private final List<ZoneInfo> zones;
    private final Map<String, List<HourRow>> rows = new HashMap<>();
    private final String datasetKind;
    private final LocalDateTime start;
    private final LocalDateTime end;
    private final LocalDateTime testStart;

    public OccupancyHistory(ParkingProperties properties) throws IOException {
        Path directory = properties.dataDir();
        ObjectMapper mapper = new ObjectMapper();
        zones = mapper.readValue(directory.resolve("zones.json").toFile(), new TypeReference<>() { });
        JsonNode metadata = mapper.readTree(directory.resolve("metadata.json").toFile());
        datasetKind = metadata.get("dataset_kind").asText();
        testStart = LocalDateTime.parse(metadata.get("test_start").asText(), CSV_TIME);
        readRows(directory.resolve("occupancy_hourly.csv"));
        List<HourRow> first = rows.get(zones.get(0).id());
        start = first.get(0).time();
        end = first.get(first.size() - 1).time();
    }

    private void readRows(Path path) throws IOException {
        try (BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            List<String> header = Arrays.asList(reader.readLine().trim().split(","));
            String line;
            while ((line = reader.readLine()) != null) {
                String[] cells = line.trim().split(",");
                rows.computeIfAbsent(cells[header.indexOf("zone")], key -> new ArrayList<>()).add(new HourRow(
                        LocalDateTime.parse(cells[header.indexOf("timestamp")], CSV_TIME),
                        (int) Double.parseDouble(cells[header.indexOf("occupied")]),
                        (int) Double.parseDouble(cells[header.indexOf("capacity")]),
                        Double.parseDouble(cells[header.indexOf("inflow")]),
                        Double.parseDouble(cells[header.indexOf("outflow")]),
                        Double.parseDouble(cells[header.indexOf("class_change")]),
                        Double.parseDouble(cells[header.indexOf("no_class_day")]),
                        Double.parseDouble(cells[header.indexOf("rain")])));
            }
        }
    }

    public HourRow at(String zone, LocalDateTime time) {
        return rows.get(zone).get((int) Duration.between(start, time).toHours());
    }

    public List<ZoneInfo> zones() {
        return zones;
    }

    public String datasetKind() {
        return datasetKind;
    }

    public LocalDateTime testStart() {
        return testStart;
    }

    /** Último instante consultable: deja 24 h por delante para comparar contra lo observado. */
    public LocalDateTime lastQueryable() {
        return end.minusHours(24);
    }
}
