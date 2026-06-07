# FireTots / FireEcology: Domain Specification
## Autonomous Fire-Regime Management with Physical Tots

### 1. Purpose

FireTots is a domain simulation for testing whether the TattleTots architecture can manage fire risk better than competent centralized alternatives. The problem is not only wildfire detection. It includes active fire detection, hotspot discovery, dry-fuel monitoring, controlled-burn monitoring, firebreak planning, autonomous drone suppression, and human-response escalation.

### 2. Core Hypothesis

A self-adaptive drone/sensor ecology can reduce total expected fire-regime cost under partial deployment, sensor churn, local heterogeneity, and changing seasonal risk better than a centralized optimizer with the same hardware and OPIR access.

The BMA is not expected to always beat centralized optimization under stable, fully deployed conditions. It should win, if it wins, through earlier useful capability during colonization, graceful degradation, local specialization, and adaptation.

### 3. Environment

A grid-based national park or preserve.

Required state variables per cell:
- Terrain: elevation, slope, aspect
- Fuel type: grass, shrub, timber, barren, water, built infrastructure
- Fuel load
- Live and dead fuel moisture
- Weather: wind, temperature, humidity, precipitation
- Firebreaks, roads, water bodies
- Human-use zones and structures
- Controlled burn plan polygons

Fire dynamics may start simple:
- Stochastic ignition from lightning or humans
- Spread rate driven by wind, slope, fuel, moisture
- Fire intensity driven by fuel load and weather
- Suppression reduces active burning area probabilistically
- Controlled burns can remain contained or escape

A simplified Rothermel/FARSITE-like model is preferred, but a calibrated cellular automaton is acceptable for v1.

### 4. Sensors and Basal Streams

All architectures receive the same possible streams.

#### 4.1 OPIR / Satellite Thermal
- Always available as a parallel backstop, not exclusive to BMA
- Cadence configurable: 1, 5, or 15 minutes
- Detects active thermal anomalies and large hotspots
- False positives and omissions should be modeled; GOES-like products can be noisy
- OPIR rescue rate is a metric: fraction of fires first detected by OPIR after local systems failed

#### 4.2 Camera Towers
- Fixed ridge or mast locations
- Line-of-sight coverage with terrain occlusion
- Smoke/fire detection; worse at night/fog/cloud
- Camera network deployment can be phased

#### 4.3 Weather Stations
- Temperature, humidity, wind, precipitation
- Used for fire weather index and spread modeling

#### 4.4 Fuel Moisture Sensors
- Slow signals; high value for dry-spot prediction

#### 4.5 Drone Platforms
Drones are not merely actuators; in the BMA architecture they are physical Tots.

### 5. Physical Tot Body Plans

A drone-Tot genome should be able to encode:
- Tank size: 0, 5, 40 gallons
- Sensor suite: RGB, thermal, multispectral, smoke/PM, weather microprobe
- Speed, endurance, battery capacity
- Battery SWAP behavior
- Water reload behavior
- Patrol/loiter/suppress/report energy allocation
- Preferred terrain/risk niche
- Escalation thresholds to other drones and to humans

Expected emergent roles:
- Scout drones: small, high endurance, information-rich, little/no water
- Strike drones: 5 or 40 gallon payload, consume scout residuals, suppress fires
- Relay drones: maintain comms in occluded terrain
- Broker/report drones or software agents: translate ecology state into human-readable alerts
- Whistleblowers: audit false detections, wasted sorties, malfunctioning scouts

### 6. Users

Initial model should include passive users with trust/habit dynamics, not office politics.

- Sector manager: attention budget for a geographic sector
- Fire operations chief: attention budget for suppression/resource allocation
- Controlled-burn manager: attention budget for prescribed burns and escape detection

Each user has:
- Attention budget per step
- Geographic or operational priorities
- Trust state for reporting agents
- Reporting habit cadence

### 7. Competing Architectures

No strawmen. Each competitor gets the same hardware availability and sensor feeds.

#### A0 Human/manual baseline
Human lookouts, patrols, public reports, conventional dispatch.

#### A1 Current camera/ML network
ALERT-style camera network + smoke/fire ML + human verification + OPIR backstop.

#### A2 Centralized drone fleet optimizer
Strong conventional competitor.
- Same drone fleet as BMA
- Same OPIR/camera/weather/fuel streams
- Global state estimator
- Centralized fire spread prediction
- Receding-horizon routing/assignment for drone patrol and suppression
- Human escalation pipeline

#### A3 Federated/edge network
Local nodes detect and escalate independently; limited cross-region fusion.

#### A4 BMA / TattleTots ecology
Self-organizing drone/sensor/reporting ecology using dual-currency metabolism.

### 8. Deployment / Colonization Scenario

Model phased deployment rather than assuming full steady state.

Representative spiral acquisition schedule:
- Months 0-6: 5 drones, 3 cameras, 4 weather stations, OPIR backstop, 50 km² pilot
- Months 6-12: 15 drones, 8 cameras, expanded sector, first autonomous suppression attempts
- Months 12-18: 30 drones, 12 cameras, fuel moisture sensors, full sector operations
- Months 18-24: expand to second sector or alter fleet composition based on results

Compare BMA and centralized optimizer under the same hardware rollout.

### 9. Metrics

- Active fire detection latency
- Hotspot detection lead time
- Dry-spot prediction precision/recall
- Controlled burn escape detection latency
- Autonomous suppression success rate
- Human crew escalation rate
- False dispatch rate
- Drone sortie cost
- Battery/water logistics cost
- Damage cost
- Surveillance cost
- Response cost
- Attention load per user
- OPIR rescue rate
- Robustness to drone/sensor failure
- Colonization curve: capability vs deployment phase

### 10. Falsification Test

BMA only matters if it reduces total expected cost:

surveillance + drone operations + response + damage + attention cost

while matching or improving detection/suppression performance against the centralized drone fleet optimizer.
