"""A4: BMA/TattleTots — decentralized self-organizing Tot ecology.

Wraps the TattleTots engine, feeding it fire-domain streams via the
``FireEcologyAdapter`` and translating Tot escalation reports back into
fire-grid detections and suppressions.
"""

from __future__ import annotations

import numpy as np
from tattletots.engine.config import SimulationConfig
from tattletots.engine.world import World

from fire_ecology.adapter.fire_adapter import FireEcologyAdapter
from fire_ecology.architectures.base import Architecture, ArchitectureResult
from fire_ecology.drones.body_plan import BodyPlan
from fire_ecology.environment.fire import FireGrid
from fire_ecology.environment.weather import WeatherState
from fire_ecology.sensors.opir import OPIRSatellite


class BMAFireEcology(Architecture):
    """BMA architecture using TattleTots as the decision-making engine.

    Each simulation step:
    1. Advance the internal ``FireEcologyAdapter`` (updates sensor streams).
    2. Feed ground truth to the ``World`` engine.
    3. Run one TattleTots engine step (agents compress, report, evolve).
    4. Map Tot escalation count to fire-grid detections.
    5. Apply suppressions with body-plan-dependent effectiveness.
    """

    def __init__(
        self,
        n_drones: int = 10,
        grid_rows: int = 20,
        grid_cols: int = 20,
        seed: int = 42,
        body_plan: BodyPlan | None = None,
        initial_population: int = 15,
    ) -> None:
        self.n_drones = n_drones
        self.body_plan = body_plan or BodyPlan.hybrid()
        self._seed = seed
        self._grid_rows = grid_rows
        self._grid_cols = grid_cols
        self._initial_population = initial_population

        self._adapter: FireEcologyAdapter | None = None
        self._world: World | None = None
        self._initialized = False

    def _init_engine(self) -> None:
        """Lazy-initialize the TattleTots engine and adapter."""
        self._adapter = FireEcologyAdapter(
            grid_rows=self._grid_rows,
            grid_cols=self._grid_cols,
            seed=self._seed,
        )

        config = SimulationConfig(
            max_population=self.n_drones * 5,
            initial_population=self._initial_population,
            seed=self._seed,
            max_steps=10000,
            mutation_rate=0.1,
        )
        self._world = World(config=config)

        for stream in self._adapter.get_streams():
            self._world.add_stream(stream)
        for user in self._adapter.get_users():
            self._world.add_user(user)

        self._world.seed_population()
        self._initialized = True

    def step(
        self,
        fire_grid: FireGrid,
        weather: WeatherState,
        opir: OPIRSatellite,
        time_step: int,
        rng: np.random.Generator,
    ) -> ArchitectureResult:
        if not self._initialized:
            self._init_engine()
        assert self._adapter is not None
        assert self._world is not None

        # 1. Advance domain simulation (updates stream data)
        self._adapter.step(time_step)

        # 2. Feed ground truth
        has_fire = len(fire_grid.active_fire_cells()) > 0
        self._world.set_ground_truth(has_fire)

        # 3. Run one TattleTots step
        record = self._world.step()

        # 4. Map Tot escalations → fire-grid detections
        #    The StepRecord gives us counts (reports_issued, correct_reports).
        #    We translate each report into a detection of the highest-intensity
        #    active fire cell, simulating the collective Tot ecology directing
        #    human attention to the worst hotspots.
        active = fire_grid.active_fire_cells()
        detections: list[tuple[int, int]] = []
        if record.reports_issued > 0 and active:
            ranked = sorted(
                active, key=lambda rc: fire_grid.fire[rc[0]][rc[1]].intensity, reverse=True
            )
            for cell in ranked[: record.reports_issued]:
                if cell not in detections:
                    detections.append(cell)

        # 5. OPIR backstop
        opir_hits = opir.scan(fire_grid, time_step, rng)
        for r, c, _conf in opir_hits:
            if (r, c) not in detections:
                detections.append((r, c))

        # 6. Suppression using body-plan effectiveness
        suppressions: list[tuple[int, int]] = []
        effectiveness = self.body_plan.suppression_effectiveness
        for r, c in detections[: self.n_drones]:
            if self.body_plan.can_suppress and fire_grid.suppress(r, c, effectiveness):
                suppressions.append((r, c))

        # 7. Cost: drone operations + TattleTots overhead
        cost = self.n_drones * 0.4
        cost += len(suppressions) * 1.5

        return ArchitectureResult(
            detections=detections,
            suppressions=suppressions,
            escalations=record.reports_issued,
            cost=cost,
        )

    @property
    def world(self) -> World | None:
        """Expose the TattleTots engine for inspection."""
        return self._world

    @property
    def living_population(self) -> int:
        """Current number of living Tots."""
        if self._world is None:
            return 0
        return self._world.living_population

    def reset(self) -> None:
        self._adapter = None
        self._world = None
        self._initialized = False
