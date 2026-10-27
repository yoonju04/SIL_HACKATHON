from integration.behavior_agent import RoadOption


def test_turn_counts_when_a_maneuver_contains_many_samples() -> None:
    from integration.urban_route import count_turns
    # Given: two left junctions and two right junctions with dense route samples.
    options = [RoadOption.LANEFOLLOW, RoadOption.LEFT, RoadOption.LEFT,
               RoadOption.LANEFOLLOW, RoadOption.RIGHT, RoadOption.RIGHT,
               RoadOption.STRAIGHT, RoadOption.LEFT, RoadOption.LANEFOLLOW,
               RoadOption.RIGHT]
    # When: counting distinct maneuvers.
    result = count_turns(options)
    # Then: samples belonging to the same maneuver do not inflate the count.
    assert result == (2, 2)
