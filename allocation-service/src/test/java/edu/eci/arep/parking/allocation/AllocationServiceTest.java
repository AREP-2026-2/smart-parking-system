package edu.eci.arep.parking.allocation;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import edu.eci.arep.parking.TestData;
import edu.eci.arep.parking.allocation.AllocationService.Assignment;
import edu.eci.arep.parking.availability.DigitalTwin;
import edu.eci.arep.parking.availability.OccupancyHistory;
import edu.eci.arep.parking.availability.ParkingLayout;
import edu.eci.arep.parking.config.ParkingProperties;
import edu.eci.arep.parking.domain.SpotSize;
import edu.eci.arep.parking.domain.Vehicle;
import edu.eci.arep.parking.prediction.ForecastGateway;
import edu.eci.arep.parking.prediction.PredictionClient;
import edu.eci.arep.parking.prediction.PredictionDtos.FeedbackRequest;
import edu.eci.arep.parking.prediction.PredictionDtos.ForecastRequest;
import edu.eci.arep.parking.prediction.PredictionDtos.ForecastResponse;
import edu.eci.arep.parking.prediction.PredictionDtos.Query;
import edu.eci.arep.parking.prediction.PredictionDtos.Result;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.web.client.ResourceAccessException;

class AllocationServiceTest {

    private static final Vehicle CAR = new Vehicle(SpotSize.MEDIUM, false);

    @TempDir
    Path directory;

    private final List<ForecastRequest> requests = new ArrayList<>();
    private Map<String, Double> predicted = Map.of("A", 0.1, "B", 0.1);
    private boolean serviceDown;

    private final PredictionClient client = new PredictionClient() {
        @Override
        public ForecastResponse forecast(ForecastRequest request) {
            if (serviceDown) {
                throw new ResourceAccessException("caído");
            }
            requests.add(request);
            return new ForecastResponse(request.queries().stream().map(query -> new Result(query.zone(),
                    query.horizonHours(), query.asOf(), predicted.get(query.zone()), "lstm", "lstm-h1-v1",
                    false, null, null)).toList());
        }

        @Override
        public void feedback(FeedbackRequest request) {
        }
    };

    private AllocationService service(int occupiedA, int occupiedB) throws IOException {
        ParkingProperties properties = TestData.create(directory, occupiedA, occupiedB);
        OccupancyHistory history = new OccupancyHistory(properties);
        twin = new DigitalTwin(history, new ParkingLayout(history));
        return new AllocationService(twin, new ForecastGateway(history, client), properties);
    }

    private DigitalTwin twin;

    @Test
    void picksNearestWhenZonesAreEquallyEmpty() throws IOException {
        Assignment result = service(0, 0).assign(CAR, 0, null, false);
        assertTrue(result.assigned());
        assertEquals("A", result.best().spot().zone());
        assertEquals(result.nearest().spot(), result.best().spot());
    }

    @Test
    void predictedSaturationMovesAssignmentToOtherZone() throws IOException {
        predicted = Map.of("A", 1.0, "B", 0.0);
        Assignment result = service(0, 0).assign(CAR, 0, null, false);
        assertEquals("B", result.best().spot().zone());
        assertEquals("A", result.nearest().spot().zone());
    }

    @Test
    void localizedCongestionMovesAssignmentToOtherZone() throws IOException {
        AllocationService service = service(0, 0);
        twin.setCongestion("A", 1.0);
        assertEquals("B", service.assign(CAR, 0, null, false).best().spot().zone());
    }

    @Test
    void horizonIncludesPlannedArrivalAndTravelTime() throws IOException {
        service(0, 0).assign(CAR, 4, null, false);
        Query query = requests.get(0).queries().get(0);
        assertTrue(query.horizonHours() > 4 && query.horizonHours() < 4.2);
        assertEquals(24, query.history().size());
        assertEquals(query.asOf(), query.history().get(23).timestamp());
    }

    @Test
    void reservedSpotIsNotAssignedAgain() throws IOException {
        AllocationService service = service(0, 0);
        Assignment first = service.assign(CAR, 0, null, true);
        Assignment second = service.assign(CAR, 0, null, true);
        assertFalse(first.best().spot().equals(second.best().spot()));
    }

    @Test
    void movingTheClockClearsReservations() throws IOException {
        AllocationService service = service(0, 0);
        Assignment first = service.assign(CAR, 0, null, true);
        twin.moveTo(TestData.TEST_START.plusHours(1));
        assertEquals(first.best().spot(), service.assign(CAR, 0, null, false).best().spot());
    }

    @Test
    void reportsNoAssignmentWhenParkingIsFull() throws IOException {
        Assignment result = service(40, 40).assign(CAR, 0, null, false);
        assertFalse(result.assigned());
        assertTrue(requests.isEmpty());
    }

    @Test
    void keepsAssigningWithBaselineWhenPredictionServiceIsDown() throws IOException {
        AllocationService service = service(10, 10);
        serviceDown = true;
        Assignment result = service.assign(CAR, 0, null, false);
        assertTrue(result.assigned());
        assertTrue(result.degraded());
        Result forecast = result.forecasts().get(0);
        assertEquals("baseline", forecast.branch());
        assertEquals("prediction_service_unavailable", forecast.reason());
        assertEquals(0.25, forecast.occupancy(), 1e-9);
    }

    @Test
    void accessibilityNeedIsServed() throws IOException {
        Assignment result = service(0, 0).assign(new Vehicle(SpotSize.MEDIUM, true), 0, null, false);
        assertTrue(result.best().spot().accessible());
        assertTrue(result.accessibilityMet());
    }

    @Test
    void clockCannotLeaveTheTestPeriod() throws IOException {
        service(0, 0);
        assertThrows(IllegalArgumentException.class, () -> twin.moveTo(TestData.START));
    }

    @Test
    void experimentComparesAllPoliciesOverSameVehicles() throws IOException {
        ParkingProperties properties = TestData.create(directory, 5, 5);
        OccupancyHistory history = new OccupancyHistory(properties);
        AllocationExperiment experiment = new AllocationExperiment(history, new ParkingLayout(history),
                new ForecastGateway(history, client));
        AllocationExperiment.Result3 result = experiment.run(TestData.WEIGHTS);
        assertEquals(4, result.policies().size());
        assertTrue(result.events() > 0);
        assertEquals(result.events() * 8, result.vehicles());
        assertEquals(0, result.degradedEvents());
    }
}
