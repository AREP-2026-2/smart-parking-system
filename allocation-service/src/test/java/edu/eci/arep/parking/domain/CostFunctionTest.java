package edu.eci.arep.parking.domain;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class CostFunctionTest {

    private static final Weights WEIGHTS = new Weights(0.25, 0.20, 0.15, 0.10, 0.10, 0.20);
    private static final ZoneContext CLEAR = new ZoneContext(0, 0);
    private static final Vehicle MEDIUM_CAR = new Vehicle(SpotSize.MEDIUM, false);

    private static Spot spot(String id, String zone, double distance, SpotSize size, boolean accessible) {
        return new Spot(id, zone, distance, size, accessible);
    }

    @Test
    void totalAppliesEveryWeightToItsTerm() {
        CostTerms terms = new CostTerms(0.1, 0.2, 0.3, 0.4, 0.5, 0.6);
        Weights weights = new Weights(1, 10, 100, 1000, 10000, 100000);
        assertEquals(0.1 + 2 + 30 + 400 + 5000 + 60000, terms.total(weights), 1e-6);
    }

    @Test
    void vehicleMustFitTheSpot() {
        assertFalse(CostFunction.eligible(spot("a", "A", 10, SpotSize.COMPACT, false), MEDIUM_CAR));
        assertTrue(CostFunction.eligible(spot("b", "A", 10, SpotSize.LARGE, false), MEDIUM_CAR));
    }

    @Test
    void accessibleSpotIsNeverGivenToWhoDoesNotNeedIt() {
        Spot accessible = spot("a", "A", 1, SpotSize.LARGE, true);
        Spot regular = spot("b", "A", 500, SpotSize.MEDIUM, false);
        assertEquals(List.of(regular), CostFunction.candidates(List.of(accessible, regular), MEDIUM_CAR));
    }

    @Test
    void accessibleSpotWinsEvenIfMuchFartherForWhoNeedsIt() {
        Spot regular = spot("a", "A", 10, SpotSize.MEDIUM, false);
        Spot accessible = spot("b", "A", 600, SpotSize.LARGE, true);
        Vehicle vehicle = new Vehicle(SpotSize.MEDIUM, true);
        List<ScoredSpot> ranked = CostFunction.rank(List.of(regular, accessible), vehicle, Map.of("A", CLEAR), WEIGHTS);
        assertEquals("b", ranked.get(0).spot().id());
    }

    @Test
    void whoNeedsAccessibilityStillGetsASpotWhenNoneIsLeft() {
        Spot regular = spot("a", "A", 10, SpotSize.MEDIUM, false);
        Vehicle vehicle = new Vehicle(SpotSize.MEDIUM, true);
        ScoredSpot best = CostFunction.rank(List.of(regular), vehicle, Map.of("A", CLEAR), WEIGHTS).get(0);
        assertEquals(1, best.terms().accessibility());
    }

    @Test
    void predictedSaturationCanOverrideProximity() {
        Spot near = spot("near", "A", 100, SpotSize.MEDIUM, false);
        Spot far = spot("far", "B", 150, SpotSize.MEDIUM, false);
        Map<String, ZoneContext> zones = Map.of("A", new ZoneContext(0, 1), "B", new ZoneContext(0, 0));
        assertEquals("far", CostFunction.rank(List.of(near, far), MEDIUM_CAR, zones, WEIGHTS).get(0).spot().id());
        assertEquals("near", CostFunction.rank(List.of(near, far), MEDIUM_CAR, zones,
                WEIGHTS.withoutSaturation()).get(0).spot().id());
    }

    @Test
    void congestionSlowsTravelAndRaisesCost() {
        Spot spot = spot("a", "A", 240, SpotSize.MEDIUM, false);
        assertEquals(2.0, CostFunction.travelMinutes(240, 0), 1e-9);
        assertEquals(6.0, CostFunction.travelMinutes(240, 1), 1e-9);
        assertTrue(CostFunction.score(spot, MEDIUM_CAR, new ZoneContext(1, 0), WEIGHTS).cost()
                > CostFunction.score(spot, MEDIUM_CAR, CLEAR, WEIGHTS).cost());
    }

    @Test
    void oversizedSpotIsPenalizedProportionally() {
        Vehicle compact = new Vehicle(SpotSize.COMPACT, false);
        assertEquals(0, CostFunction.score(spot("a", "A", 1, SpotSize.COMPACT, false), compact, CLEAR, WEIGHTS).terms().inadequacy());
        assertEquals(0.5, CostFunction.score(spot("b", "A", 1, SpotSize.MEDIUM, false), compact, CLEAR, WEIGHTS).terms().inadequacy());
        assertEquals(1, CostFunction.score(spot("c", "A", 1, SpotSize.LARGE, false), compact, CLEAR, WEIGHTS).terms().inadequacy());
    }

    @Test
    void nearestIgnoresCostAndRespectsEligibility() {
        Spot tooSmall = spot("a", "A", 1, SpotSize.COMPACT, false);
        Spot valid = spot("b", "B", 50, SpotSize.MEDIUM, false);
        assertEquals(valid, CostFunction.nearest(List.of(tooSmall, valid), MEDIUM_CAR).orElseThrow());
    }
}
