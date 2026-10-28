"""Native CARLA geometry with in-memory actors; never connects to a server."""
from fnmatch import fnmatch
from pathlib import Path
import sys
from typing import List

import carla

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'third_party' / 'carla_0915'))

OPENDRIVE = '''<OpenDRIVE>
<header revMajor="1" revMinor="4" name="behavior_fixture" version="1.00" date="" north="0" south="0" east="300" west="0"><geoReference>+proj=tmerc +lat_0=42 +lon_0=2 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs</geoReference></header>
<road name="straight" length="300" id="1" junction="-1">
<link><successor elementType="road" elementId="2" contactPoint="start" /></link>
<type s="0" type="town"><speed max="50" unit="km/h" /></type>
<planView><geometry s="0" x="0" y="0" hdg="0" length="300"><line /></geometry></planView>
<elevationProfile><elevation s="0" a="0" b="0" c="0" d="0" /></elevationProfile>
<lateralProfile />
<lanes><laneSection s="0"><center><lane id="0" type="none" level="false"><roadMark sOffset="0" type="solid" weight="standard" color="white" width="0.15" laneChange="none" /></lane></center>
<right><lane id="-1" type="driving" level="false"><link><successor id="-1" /></link><width sOffset="0" a="3.5" b="0" c="0" d="0" /><roadMark sOffset="0" type="solid" weight="standard" color="white" width="0.15" laneChange="none" /></lane></right>
</laneSection></lanes><objects /><signals />
</road><road name="straight" length="300" id="2" junction="-1">
<link><predecessor elementType="road" elementId="1" contactPoint="end" /></link><type s="0" type="town"><speed max="50" unit="km/h" /></type>
<planView><geometry s="0" x="300" y="0" hdg="0" length="300"><line /></geometry></planView>
<elevationProfile><elevation s="0" a="0" b="0" c="0" d="0" /></elevationProfile>
<lateralProfile />
<lanes><laneSection s="0"><center><lane id="0" type="none" level="false"><roadMark sOffset="0" type="solid" weight="standard" color="white" width="0.15" laneChange="none" /></lane></center>
<right><lane id="-1" type="driving" level="false"><link><predecessor id="-1" /></link><width sOffset="0" a="3.5" b="0" c="0" d="0" /><roadMark sOffset="0" type="solid" weight="standard" color="white" width="0.15" laneChange="none" /></lane></right>
</laneSection></lanes><objects /><signals />
</road></OpenDRIVE>'''

class Actors(List['Actor']):
    def filter(self, pattern: str) -> 'Actors':
        return Actors(actor for actor in self if fnmatch(actor.type_id, pattern))


class World:
    """Mutable actor registry around a real, immutable road map."""
    def __init__(self) -> None:
        self.map = carla.Map('behavior_fixture', OPENDRIVE)
        self.actors = Actors()

    def get_map(self) -> carla.Map:
        return self.map

    def get_actors(self) -> Actors:
        return self.actors


class Actor:
    """Mutable test pose and speed with native copy-on-read geometry."""
    def __init__(self, world: World, type_id: str, x: float) -> None:
        self.world = world
        self.id = len(world.actors) + 1
        self.type_id = type_id
        self.x = x
        self.speed = 0.0
        self.speed_limit = 50.0
        self.bounding_box = carla.BoundingBox(carla.Location(), carla.Vector3D(2.0, 0.9, 0.8))
        self.trigger_volume = carla.BoundingBox(carla.Location(), carla.Vector3D(1.0, 1.0, 1.0))
        self.state = carla.TrafficLightState.Red
        world.actors.append(self)

    def get_world(self) -> World:
        return self.world

    def get_location(self) -> carla.Location:
        return carla.Location(x=self.x, y=1.75)

    def get_transform(self) -> carla.Transform:
        return carla.Transform(self.get_location(), carla.Rotation())

    def get_velocity(self) -> carla.Vector3D:
        return carla.Vector3D(self.speed, 0.0, 0.0)

    def get_control(self) -> carla.VehicleControl:
        return carla.VehicleControl()

    def get_speed_limit(self) -> float:
        return self.speed_limit

    def apply_control(self, control: carla.VehicleControl) -> None:
        raise AssertionError('Behavior planning must never apply vehicle controls')


def make_ego() -> Actor:
    return Actor(World(), 'vehicle.ego', 20.0)


def test_native_map_fixture() -> None:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    # Given: native CARLA math and a native OpenDRIVE map, without a server.
    ego = make_ego()
    # When: the official global planner traces a destination.
    planner = GlobalRoutePlanner(ego.world.map, 2.0)
    route = planner.trace_route(ego.get_location(), carla.Location(140.0, 1.75, 0.0))
    # Then: the route advances along the expected native driving lane.
    assert len(route) > 40
    assert route[0][0].lane_id == -1
    assert route[-1][0].transform.location.x >= 138.0
    assert abs(route[0][0].transform.location.y - 1.75) < 0.001


if __name__ == '__main__':
    test_native_map_fixture()
    print('PASS native CARLA map and official GlobalRoutePlanner, server-free')
