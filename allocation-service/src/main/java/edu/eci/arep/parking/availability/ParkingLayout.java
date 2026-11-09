package edu.eci.arep.parking.availability;

import edu.eci.arep.parking.availability.OccupancyHistory.ZoneInfo;
import edu.eci.arep.parking.domain.Spot;
import edu.eci.arep.parking.domain.SpotSize;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import org.springframework.stereotype.Component;

/** Plano simplificado: cada zona a una distancia de la entrada, con filas de 20 espacios. */
@Component
public class ParkingLayout {

    private static final double[] ZONE_DISTANCE = {40, 130, 230, 320, 430};
    private static final int ROW_LENGTH = 20;
    private static final int ACCESSIBLE_EVERY = 25;

    private final List<Spot> spots = new ArrayList<>();
    private final Map<String, List<Spot>> byZone = new HashMap<>();

    public ParkingLayout(OccupancyHistory history) {
        List<ZoneInfo> zones = history.zones();
        for (int z = 0; z < zones.size(); z++) {
            ZoneInfo zone = zones.get(z);
            double base = ZONE_DISTANCE[Math.min(z, ZONE_DISTANCE.length - 1)];
            List<Spot> zoneSpots = new ArrayList<>();
            for (int i = 0; i < zone.capacity(); i++) {
                double distance = base + (i % ROW_LENGTH) * 6 + (double) (i / ROW_LENGTH) * 12;
                boolean accessible = i % ACCESSIBLE_EVERY == 0;
                zoneSpots.add(new Spot("%s-%03d".formatted(zone.id(), i + 1), zone.id(), distance,
                        accessible ? SpotSize.LARGE : sizeFor(i), accessible));
            }
            byZone.put(zone.id(), zoneSpots);
            spots.addAll(zoneSpots);
        }
    }

    private static SpotSize sizeFor(int index) {
        int slot = index % 10;
        if (slot == 9) {
            return SpotSize.LARGE;
        }
        return slot < 6 ? SpotSize.MEDIUM : SpotSize.COMPACT;
    }

    public List<Spot> spots() {
        return spots;
    }

    public List<Spot> spotsOf(String zone) {
        return byZone.get(zone);
    }

    /** Los datos solo traen conteos: qué espacios concretos están ocupados se fija de forma reproducible. */
    public Set<String> occupiedSpots(String zone, int count, LocalDateTime time) {
        List<Spot> shuffled = new ArrayList<>(byZone.get(zone));
        Collections.shuffle(shuffled, new Random(zone.hashCode() * 31L + time.hashCode()));
        Set<String> occupied = new HashSet<>();
        shuffled.stream().limit(count).forEach(spot -> occupied.add(spot.id()));
        return occupied;
    }
}
