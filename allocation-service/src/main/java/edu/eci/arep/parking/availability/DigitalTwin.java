package edu.eci.arep.parking.availability;

import edu.eci.arep.parking.availability.OccupancyHistory.HourRow;
import edu.eci.arep.parking.availability.OccupancyHistory.ZoneInfo;
import edu.eci.arep.parking.domain.Spot;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.springframework.stereotype.Service;

/** Representación digital del parqueadero (Módulo 3) sobre un reloj de reproducción. */
@Service
public class DigitalTwin {

    /** Entradas por hora, relativas a la capacidad, que equivalen a congestión máxima. */
    private static final double FULL_CONGESTION_INFLOW = 0.5;

    public record SpotState(Spot spot, boolean occupied, boolean reserved) {
    }

    public record ZoneState(String id, String name, int capacity, int occupied, double occupancy, double congestion) {
    }

    private final OccupancyHistory history;
    private final ParkingLayout layout;
    private final Set<String> reserved = new HashSet<>();
    private final Map<String, Double> congestionOverrides = new HashMap<>();
    private LocalDateTime asOf;

    public DigitalTwin(OccupancyHistory history, ParkingLayout layout) {
        this.history = history;
        this.layout = layout;
        this.asOf = history.testStart();
    }

    public synchronized LocalDateTime asOf() {
        return asOf;
    }

    /** Mueve el reloj; las reservas y las congestiones inyectadas no sobreviven al cambio. */
    public synchronized void moveTo(LocalDateTime time) {
        LocalDateTime hour = time.withMinute(0).withSecond(0).withNano(0);
        if (hour.isBefore(history.testStart()) || hour.isAfter(history.lastQueryable())) {
            throw new IllegalArgumentException("El reloj debe estar entre %s y %s"
                    .formatted(history.testStart(), history.lastQueryable()));
        }
        asOf = hour;
        reset();
    }

    public synchronized void reset() {
        reserved.clear();
        congestionOverrides.clear();
    }

    public synchronized void setCongestion(String zone, double value) {
        congestionOverrides.put(zone, value);
    }

    public synchronized boolean reserve(String spotId) {
        return reserved.add(spotId);
    }

    public synchronized List<SpotState> spots() {
        Map<String, Set<String>> occupied = new HashMap<>();
        for (ZoneInfo zone : history.zones()) {
            occupied.put(zone.id(), layout.occupiedSpots(zone.id(), history.at(zone.id(), asOf).occupied(), asOf));
        }
        List<SpotState> states = new ArrayList<>();
        for (Spot spot : layout.spots()) {
            states.add(new SpotState(spot, occupied.get(spot.zone()).contains(spot.id()), reserved.contains(spot.id())));
        }
        return states;
    }

    public List<Spot> freeSpots() {
        return spots().stream().filter(state -> !state.occupied() && !state.reserved()).map(SpotState::spot).toList();
    }

    public synchronized List<ZoneState> zones() {
        Map<String, Long> reservedByZone = new HashMap<>();
        for (SpotState state : spots()) {
            if (state.reserved() && !state.occupied()) {
                reservedByZone.merge(state.spot().zone(), 1L, Long::sum);
            }
        }
        return history.zones().stream().map(zone -> {
            HourRow row = history.at(zone.id(), asOf);
            int occupied = row.occupied() + reservedByZone.getOrDefault(zone.id(), 0L).intValue();
            double measured = Math.min(1, row.inflow() / row.capacity() / FULL_CONGESTION_INFLOW);
            return new ZoneState(zone.id(), zone.name(), zone.capacity(), occupied,
                    (double) occupied / zone.capacity(),
                    congestionOverrides.getOrDefault(zone.id(), measured));
        }).toList();
    }
}
