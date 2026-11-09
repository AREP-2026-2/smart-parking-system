package edu.eci.arep.parking.domain;

public enum SpotSize {
    COMPACT, MEDIUM, LARGE;

    public boolean fits(SpotSize vehicle) {
        return ordinal() >= vehicle.ordinal();
    }
}
