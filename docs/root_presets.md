# Root presets and LAI orientation values – sources

## How ENVI-met describes roots

Root zone = body of revolution below the trunk with depth and diameter, split into 10 layers of equal
thickness. Per layer: root area density `RAD` [m²/m³] and horizontal extent as a fraction of the
radius. The plugin generates these from: shape, depth, radius (absolute or factor × crown radius),
vertical distribution and either a mean RAD or a total root area.

## Presets

### Vertical distribution

| Preset | What it sets | Source | Checked how |
|---|---|---|---|
| Forest, cool-temperate | logistic distribution, D50 = 21 cm, D95 = 104 cm | Schenk & Jackson (2002), Table 4: cool-temperate forest including plantations, c = −1.835 |
| Forest, warm-temperate | logistic distribution, D50 = 23 cm, D95 = 121 cm | Schenk & Jackson (2002), Table 4, c = −1.757 |
| Forest, boreal | logistic distribution, D50 = 12 cm, D95 = 58 cm | Schenk & Jackson (2002), Table 4, c = −1.880 |
| Deciduous trees, temperate | exponential distribution, β = 0.966 | Jackson et al. (1996), Table 1: temperate deciduous forest, 65 % of root biomass in the upper 30 cm |
| Conifers, temperate | exponential distribution, β = 0.976 | Jackson et al. (1996), Table 1: temperate coniferous forest, 52 % in the upper 30 cm |

**Logistic model** (Schenk & Jackson 2002, Eq. 1): cumulative root fraction above depth D

    r(D) = 1 / (1 + (D / D50)^c)

D50 and D95 are the depths above which 50 % and 95 % of the roots lie. The plugin asks for these two
depths and derives the shape parameter from them, `c = ln(1/19) / ln(D95 / D50)`. For the three table
rows this reproduces the published c within rounding (−1.84, −1.77, −1.87). The published values are
means of profiles extrapolated to at most 3 m depth.

**Exponential model** (Gale & Grigal 1987, as used by Jackson et al. 1996): `Y = 1 − β^d` with soil
depth `d` in **cm**; larger β = deeper roots.

Both models are evaluated for the ten layers of the chosen root depth and normalised (the part of
the theoretical profile below the chosen depth is cut off).

Check against the data set behind Schenk & Jackson (ORNL DAAC data set 660, 564 profiles,
`root_profiles_D50D95.csv`): means of the extrapolated depths are D50 = 0.22 m / D95 = 1.13 m for
temperate deciduous broadleaved forest (n = 34) and 0.23 m / 1.12 m for temperate needle-leaved
forest (n = 27). The two groups do not differ, so there is one forest preset per climate zone and
no split into deciduous and coniferous trees. (Jackson et al. 1996 did find a difference with fewer
profiles; both presets are offered.)

### Shape and depth by species

| Preset | What it sets | Source | Checked how |
|---|---|---|---|
| European beech (*Fagus sylvatica*) | heart root, depth 1.25 m, cool-temperate logistic distribution | type: Büsgen et al. (1929) in Crow (2005); depth: Eichhorn (1992) in Puhe (2003) | full text of Crow and Puhe; the primary sources were not seen |
| Oak (*Quercus robur*, *Q. petraea*) | tap root, depth 1.26 m, same distribution | as above | as above |
| Scots pine (*Pinus sylvestris*) | tap root, depth 1.47 m, same distribution | as above | as above |
| Norway spruce (*Picea abies*) | surface (plate) root system, depth 0.96 m, same distribution | as above | as above |
| Lime, Norway maple, birch, larch | heart root system – shape only | Büsgen et al. (1929) in Crow (2005) | full text of Crow |
| Ash, aspen | surface root system – shape only | Büsgen et al. (1929) in Crow (2005) | full text of Crow |

* **Root system types.** Crow (2005) quotes the three types of Büsgen et al. (1929): tap root (English
  oak, Scots pine, silver fir), heart root (birch, beech, larch, lime, Norway maple) and surface root
  systems (ash, aspen, Norway spruce, white pine). He stresses that this is a broad generalisation, that
  soil conditions often outweigh the species, and that the type says nothing about the depth. The
  outlines the plugin draws for these types are schematic.
* **Depths.** Puhe (2003) reports from Eichhorn (1992) the mean maximum coarse-root depth in the
  central root stock of mature forest trees on soils without waterlogging: Norway spruce 0.96 m,
  European beech 1.25 m, pedunculate and sessile oak 1.26 m, Scots pine 1.47 m. A second study there
  (Zoth & Block 1992, root balls of wind-thrown trees) gives 1.09 m for spruce and beech, 1.13 m for
  oak and 1.25 m for pine. No such value was found for lime, maple, birch, larch, ash or aspen, so
  these presets leave the depth untouched.
* **General orientation for the depth** (tooltip of the depth field): roots deeper than 2 m are
  uncommon and 80–90 % lie in the upper 60 cm (Crow 2005); about 99 % lie in the upper metre (Perry
  1982). The *maximum* rooting depth can be much greater on deep, loose soils: on average 2.9 ± 0.2 m
  for temperate deciduous and 3.9 ± 0.4 m for temperate coniferous forest (Canadell et al. 1996), and
  far more for single trees (Stone & Kalisz 1991).
* For street trees the built soil decides: impenetrable layers and paving often limit depth and
  spread much more than the species (Day et al. 2010).

### Horizontal extent

| Preset | What it sets | Source |
|---|---|---|---|
| Woody plants: root system 2.3 × crown width | radius factor 2.3 | Tumber-Dávila et al. (2022): mean ratio shoot width : root system width of woody plants = 0.44 ("more than twice their shoot widths"), almost independent of aridity |
| Young open-grown trees: root spread 2.9 × crown | radius factor 2.9 | Gilman (1988): root system diameter of young nursery trees averaged 2.9 × canopy diameter, from 1.68 (*Fraxinus pennsylvanica*) to 3.77 (*Magnolia grandiflora*) |

Day et al. (2010) add that all studies relating crown and root spread used trees below 18 cm trunk
diameter and that crown spread is a poor predictor for mature trees; for young trees the root system
radius grows by about 38 cm per cm of trunk diameter and levels off at 25–30 cm trunk diameter.
Perry (1982) gives a root area of four to seven times the crown projection area, i.e. a radius factor
of 2 to 2.6. The plugin has no trunk diameter, so only the crown-based factors are offered.

### Generic forms and ENVI-met default

| Preset | What it sets |
|---|---|---|
| Generic: cylinder / half ellipsoid / heart / tap / plate | shape and a simple vertical distribution |
| ENVI-met default (new Albero tree) | depth 4 m, diameter 10 m, RAD 0.1 m²/m³ homogeneous |

## LAI orientation values

Shown as a hint next to the target LAI (no limit is enforced):

| Species | max. summer LAI | Source |
|---|---|---|
| *Tilia cordata* | 4.7 (n = 341, highest of 15 species) | Moser-Reischl et al. (2025) |
| *Acer platanoides* | 3.4 (n = 62) | Moser-Reischl et al. (2025) |
| *Gleditsia triacanthos* | 2.4 (lowest of 15 species) | Moser-Reischl et al. (2025) |

## References

* Bolte, A., Rahmann, T., Kuhr, M., Pogoda, P., Murach, D., v. Gadow, K. (2004): Relationships between
  tree dimension and coarse root biomass in mixed stands of European beech (*Fagus sylvatica* L.) and
  Norway spruce (*Picea abies* [L.] Karst.). Plant and Soil 264: 1–11.
* Bolte, A., Villanueva, I. (2006): Interspecific competition impacts on the morphology and
  distribution of fine roots in European beech (*Fagus sylvatica* L.) and Norway spruce (*Picea abies*
  (L.) Karst.). European Journal of Forest Research 125: 15–26.
* Canadell, J., Jackson, R.B., Ehleringer, J.R., Mooney, H.A., Sala, O.E., Schulze, E.-D. (1996):
  Maximum rooting depth of vegetation types at the global scale. Oecologia 108: 583–595.
* Crow, P. (2005): The influence of soils and species on tree root depth. Forestry Commission
  Information Note 78, Edinburgh.
* Day, S.D., Wiseman, P.E., Dickinson, S.B., Harris, J.R. (2010): Contemporary concepts of root system
  architecture of urban trees. Arboriculture & Urban Forestry 36(4): 149–159.
* Finér, L., Helmisaari, H.-S., Lõhmus, K. et al. (2007): Variation in fine root biomass of three
  European tree species: beech (*Fagus sylvatica* L.), Norway spruce (*Picea abies* L. Karst.), and
  Scots pine (*Pinus sylvestris* L.). Plant Biosystems 141(3): 394–405.
* Gale, M.R., Grigal, D.F. (1987): Vertical root distributions of northern tree species in relation to
  successional status. Canadian Journal of Forest Research 17: 829–834.
* Gilman, E.F. (1988): Predicting root spread from trunk diameter and branch spread. Journal of
  Arboriculture 14(4): 85–89.
* Jackson, R.B., Canadell, J., Ehleringer, J.R., Mooney, H.A., Sala, O.E., Schulze, E.-D. (1996): A
  global analysis of root distributions for terrestrial biomes. Oecologia 108: 389–411.
* Kalliokoski, T., Sievänen, R., Nygren, P. (2010): Tree roots as self-similar branching structures:
  axis differentiation and segment tapering in coarse roots of three boreal forest tree species.
  Trees 24: 219–236.
* Moser-Reischl, A., Franceschi, E., Rahman, M.A. et al. (2025): Spatial and temporal dynamics of the
  leaf area index (LAI) of selected tree species in urban environments. Urban Forestry & Urban
  Greening 107: 128795.
* Perry, T.O. (1982): The ecology of tree roots and the practical significance thereof. Journal of
  Arboriculture 8(8): 197–211.
* Puhe, J. (2003): Growth and development of the root system of Norway spruce (*Picea abies*) in
  forest stands – a review. Forest Ecology and Management 175: 253–273.
* Schenk, H.J., Jackson, R.B. (2002): The global biogeography of roots. Ecological Monographs 72(3):
  311–328.
* Schenk, H.J., Jackson, R.B. (2002): Rooting depths, lateral root spreads and below-ground/above-ground
  allometries of plants in water-limited ecosystems. Journal of Ecology 90: 480–494.
* Schenk, H.J., Jackson, R.B. (2003): Global distribution of root profiles in terrestrial ecosystems.
  Data set. ORNL DAAC, Oak Ridge, Tennessee. https://doi.org/10.3334/ORNLDAAC/660
* Stone, E.L., Kalisz, P.J. (1991): On the maximum extent of tree roots. Forest Ecology and Management
  46: 59–102.
* Tumber-Dávila, S.J., Schenk, H.J., Du, E., Jackson, R.B. (2022): Plant sizes and shapes above and
  belowground and their interactions with climate. New Phytologist 235: 1032–1056.
* Zhang, C., Stratopoulos, L.M.F., Pretzsch, H., Rötzer, T. (2019): How do *Tilia cordata* Greenspire
  trees cope with drought stress regarding their biomass allocation and ecosystem services? Forests
  10: 676.
* Cited through the works above, not seen: Büsgen, M., Münch, E., Thomson, T. (1929): The structure and
  life of forest trees. Chapman and Hall, London. – Eichhorn, J. (1992) and Zoth, R., Block, J. (1992)
  as cited in Puhe (2003).
