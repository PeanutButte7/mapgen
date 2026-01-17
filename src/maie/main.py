from maie.playground import Playground2D
from maie.worlds.wfc_world import WfcWorld, WfcWorldConfig
from maie.worlds.uuworld import UUWorld, UUWorldConfig
from maie.worlds.poisson_world import PoissonWorld, PoissonWorldConfig


def main():
    world = UUWorld(UUWorldConfig())
    # world = PoissonWorld(PoissonWorldConfig())
    # cfg = WfcWorldConfig()
    # cfg.tileset = "city"
    # world = WfcWorld(cfg)
    pg = Playground2D(world)
    pg.run()


if __name__ == "__main__":
    main()
