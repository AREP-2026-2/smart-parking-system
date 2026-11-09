package edu.eci.arep.parking.api;

import edu.eci.arep.parking.availability.DigitalTwin;
import edu.eci.arep.parking.availability.OccupancyHistory;
import edu.eci.arep.parking.domain.SpotSize;
import java.time.DayOfWeek;
import java.time.LocalDateTime;
import java.util.function.Predicate;
import org.springframework.stereotype.Service;

/** Prepara los escenarios A-D de la Sección 6 buscando instantes del periodo de prueba. */
@Service
public class ScenarioService {

    public record Scenario(String id, String title, String description, LocalDateTime asOf,
                           SpotSize size, boolean needsAccessible, double arrivalInHours) {
    }

    private final OccupancyHistory history;
    private final DigitalTwin twin;

    public ScenarioService(OccupancyHistory history, DigitalTwin twin) {
        this.history = history;
        this.twin = twin;
    }

    public Scenario load(String id) {
        Scenario scenario = switch (id) {
            case "A" -> new Scenario("A", "Baja demanda",
                    "Domingo: pocos vehículos y asignación directa. El pronóstico largo anticipa la reanudación.",
                    find(this::isSunday, 18), SpotSize.MEDIUM, false, 0);
            case "B" -> new Scenario("B", "Cambio de clases",
                    "Planeación con cuatro horas de anticipación para la llegada del bloque de las 10:00.",
                    find(this::isClassDay, 6), SpotSize.MEDIUM, false, 4);
            case "C" -> new Scenario("C", "Congestión localizada",
                    "La vía de acceso a la Zona A está congestionada: gana una zona más lejana.",
                    find(this::isClassDay, 11), SpotSize.MEDIUM, false, 0);
            case "D" -> new Scenario("D", "Necesidad especial",
                    "El vehículo requiere un espacio accesible.",
                    find(this::isClassDay, 10), SpotSize.MEDIUM, true, 0);
            default -> throw new IllegalArgumentException("Escenario desconocido: " + id);
        };
        twin.moveTo(scenario.asOf());
        if ("C".equals(id)) {
            twin.setCongestion("A", 1.0);
        }
        return scenario;
    }

    private boolean isSunday(LocalDateTime day) {
        return day.getDayOfWeek() == DayOfWeek.SUNDAY;
    }

    private boolean isClassDay(LocalDateTime day) {
        DayOfWeek weekday = day.getDayOfWeek();
        return weekday.compareTo(DayOfWeek.TUESDAY) >= 0 && weekday.compareTo(DayOfWeek.THURSDAY) <= 0
                && history.at(history.zones().get(0).id(), day).noClassDay() == 0;
    }

    private LocalDateTime find(Predicate<LocalDateTime> condition, int hour) {
        LocalDateTime day = history.testStart().toLocalDate().atStartOfDay();
        for (; !day.isAfter(history.lastQueryable()); day = day.plusDays(1)) {
            LocalDateTime moment = day.withHour(hour);
            if (condition.test(day) && !moment.isBefore(history.testStart()) && !moment.isAfter(history.lastQueryable())) {
                return moment;
            }
        }
        throw new IllegalStateException("El periodo de prueba no contiene un día adecuado");
    }
}
