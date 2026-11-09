package edu.eci.arep.parking.domain;

public record ScoredSpot(Spot spot, CostTerms terms, double cost, double travelMinutes) {
}
