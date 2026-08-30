---
name: searching-literature-evidence
description: Search the peer-reviewed literature with the Consensus MCP server to source a FireEcology parameter — rate of spread by fuel model, wind and slope factors, fuel moisture, ignition density, and camera/thermal/OPIR detection performance — including query construction, filter discipline, and how a hit becomes a provenance comment with an evidence grade. Use whenever a physical or sensor constant needs a citation, or when asked what the literature says about a mechanism.
---

# Searching the Literature (Consensus MCP)

The `consensus` MCP server has one tool, `search`, over ~220M papers
(Semantic Scholar, PubMed, Scopus, ArXiv). It returns title, authors, year,
journal, citation count, DOI, a Consensus URL, and the abstract.

```
mcp_tool(command="call_tool", server="consensus", tool_name="search",
         tool_args='{"query": "rate of spread fuel model grass shrub timber litter"}')
```

Run `mcp_tool(command="list_tools", server="consensus")` for the current
parameter list before using an unfamiliar filter.

## Query construction

Query in the vocabulary of the paper you want, not the question you have. Fire
behaviour papers name the quantity and the fuel:

- Good: `head fire rate of spread grass fuel model wind adjustment factor`
- Weak: `how fast does fire spread`

Quantities this repo needs sourced, and the words that find them:

- Spread — `rate of spread`, `Rothermel`, `fuel model`, `head fire`,
  `experimental burn`, `m/min`.
- Fuel — `fuel load t/ha`, `surface-area-to-volume ratio`,
  `moisture of extinction`, `live/dead fuel moisture`.
- Environment — `wind adjustment factor`, `slope correction`,
  `fire weather index`, `Haines`, `ERC`.
- Ignition — `lightning ignition density per km2`, `human-caused ignition
  rate`, `probability of ignition`.
- Intensity and duration — `fireline intensity kW/m`, `residence time`,
  `smouldering`, `fuel consumption fraction`.
- Detection — `active fire detection probability`, `MODIS`, `VIIRS`,
  `minimum detectable fire size`, `false alarm rate`, `omission/commission
  error`, `thermal camera detection range`.

Search for the mechanism, then separately for the number. The paper
establishing that slope matters is not the one that fitted the multiplier.

## Filter discipline

Default to **no filters**; every filter silently removes evidence. Specific to
this repo:

- `medical_mode=true` and `human=true` are meaningless here and will discard
  the entire fire-science and remote-sensing literature.
- `study_types`, `controlled` and `sample_size_min` describe clinical designs;
  an experimental-burn campaign is none of them.
- `domain="env,eng,geog,agri"` is the useful narrowing.
- Do **not** set `year_min`. The canonical spread model is Rothermel (1972) and
  the standard fuel models are Anderson (1982); a recency filter removes the
  primary sources and leaves you with reviews that cite them.
- `sjr_max=1` gives Q1 only; never reach for `sjr_min`, which *excludes* the top
  tiers. Note that much of the foundational work is in USDA Forest Service
  research papers, which may not be indexed at all — a search returning nothing
  is not evidence that nothing was measured.

Filters reorder as well as remove: the top hit for the same query changes when
`domain` and `year_min` are set. Re-run a promising query without filters before
calling any value *the* measurement.

## Result handling

- Default page returns 20 papers; `page_size` narrows it (5 works). `page=1`
  returns a genuinely different set on this organisation's plan, so paginate
  when the first page is all reviews.
- Twenty abstracts overflow the tool result. The output is truncated and the
  full text written to a file named in the truncation notice — **read that
  file**. Items 15-20 are frequently the measurement papers, because reviews
  rank higher.
- Spread rates and detection curves live in tables and figures. Open the DOI
  when the constant matters.
- Consensus asks for numbered inline citations with hyperlinked titles and the
  exact URLs it returned. Preserve the DOI when it gives one.

## Most of this model's constants are dimensionless — record the derivation

The literature reports absolute quantities: metres per minute, kW/m, per cent
moisture. This model uses relative multipliers:

```python
_SPREAD_RATES: dict[FuelType, float] = {
    FuelType.GRASS: 1.0,
    FuelType.SHRUB: 0.6,
    FuelType.TIMBER: 0.3,
    FuelType.BARREN: 0.0,
}
```

A relative constant is a **derived** quantity, so both the measurement and the
normalisation must be visible at the definition — otherwise the next reader
cannot tell whether 0.6 came from a paper or from a guess:

```python
_SPREAD_RATES: dict[FuelType, float] = {
    # Normalised to grass. Head-fire rates under comparable wind and moisture:
    # grass <a> m/min, shrub <b>, timber litter <c> (<Author> et al. <year>,
    # <journal>, DOI: <doi>). Grade B: fuel models standing in for these four
    # coarse classes; the ratios, not the absolute rates, are what this model
    # uses.
    FuelType.GRASS: 1.0,
```

The same applies to `slope_factor = 1.0 + 0.02 * slope` and
`wind_factor = 1.0 + 0.05 * wind_speed`: what is sourced is a measured response,
and the linearisation is a modelling choice on top of it. Record which is which,
and the range of slope and wind over which the linear form was fitted — outside
that range it is an extrapolation, not a citation.

State **what was measured**, **in what fuel and weather**, the value with its
range, and author + year + journal + DOI. Then grade it:

- **A** — direct measurement of this quantity in this fuel type and setting.
- **B** — an analogous fuel model, a different region, or a laboratory burn
  standing in for the field.
- **C** — inferred, estimated, or a declared assumption.

If no source exists, say so explicitly rather than inventing a plausible number.
A declared Grade C with a sensitivity sweep is honest; a fabricated citation is
not.

## Sensor performance is a measured quantity too

`false_positive_rate = 0.02`, `miss_rate = 0.1` and
`min_intensity_threshold = 0.2` in `src/fire_ecology/sensors/opir.py` are
detection-performance claims, and the remote-sensing literature measures them —
usually as a function of fire size and radiative power, not as a scalar. When
you source them, record the fire size the probability applies to. A miss rate
quoted without a size threshold is not a measurement of anything.

Because OPIR is the backstop available to every architecture, its performance
sets the floor for all of them: an optimistic uncited miss rate flatters every
architecture at once, which is precisely the kind of error that survives an
architecture comparison unnoticed.

## What this search must never be used for

Do not search for a value that makes an architecture comparison, ablation, or
rebaseline come out right — `docs/fire_sensor_rebaseline.md`,
`docs/opir_ablation_grounded_access.md`, or a detection-latency headline. Those
results are only informative if the fire physics and sensor performance were
sourced independently; screening candidate papers by which value helps converts
a measurement of the architecture into a measurement of the search.

Fix the query and the filters from the definition of the quantity, before
looking at what the run needs. If several papers measure it, take a stated
central value or the midpoint of the range and say which — not the end that
helps. If a sourced constant makes an architecture look worse, that is a result:
report it.
