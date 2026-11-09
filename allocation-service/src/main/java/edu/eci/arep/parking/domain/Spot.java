package edu.eci.arep.parking.domain;

public record Spot(String id, String zone, double distanceMeters, SpotSize size, boolean accessible) {
}
