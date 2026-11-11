# presets.py
# Presets with literature references (roots, LAI orientation values)
from . import roots

SRC_JACKSON = ('Jackson, R.B., Canadell, J., Ehleringer, J.R., Mooney, H.A., Sala, O.E., Schulze, E.-D. (1996): '
               'A global analysis of root distributions for terrestrial biomes. Oecologia 108: 389-411 (Table 1).')
SRC_GALE = ('Gale, M.R., Grigal, D.F. (1987): Vertical root distributions of northern tree species in relation '
            'to successional status. Can. J. For. Res. 17: 829-834.')
SRC_GILMAN = ('Gilman, E.F. (1988): Predicting root spread from trunk diameter and branch spread. '
              'Journal of Arboriculture 14(4): 85-89.')
SRC_DAY = ('Day, S.D., Wiseman, P.E., Dickinson, S.B., Harris, J.R. (2010): Contemporary concepts of root system '
           'architecture of urban trees. Arboriculture & Urban Forestry 36(4): 149-159.')
SRC_SCHENK = ('Schenk, H.J., Jackson, R.B. (2002): The global biogeography of roots. '
              'Ecological Monographs 72(3): 311-328 (Table 4).')
SRC_TUMBER = ('Tumber-Davila, S.J., Schenk, H.J., Du, E., Jackson, R.B. (2022): Plant sizes and shapes above and '
              'belowground and their interactions with climate. New Phytologist 235: 1032-1056.')
SRC_CROW = ('Crow, P. (2005): The influence of soils and species on tree root depth. '
            'Forestry Commission Information Note 78, Edinburgh.')
SRC_PUHE = ('Puhe, J. (2003): Growth and development of the root system of Norway spruce (Picea abies) in forest '
            'stands - a review. Forest Ecology and Management 175: 253-273.')
ROOT_TYPE_NOTE = ('Root system type after Buesgen et al. (1929) as cited in ' + SRC_CROW + ' Only a broad '
                  'generalisation: soil conditions often outweigh the species; the outline itself is schematic.')
DEPTH_NOTE = ('Depth = mean maximum coarse-root depth of mature forest trees on soils without waterlogging '
              '(Eichhorn 1992, cited in ' + SRC_PUHE + ').')
LOGISTIC_NOTE = ('Vertical distribution: logistic model r(D) = 1 / (1 + (D / D50)^c) for cool-temperate forest, '
                 'D50 = 21 cm, D95 = 104 cm. ' + SRC_SCHENK)
SRC_ENVIMET = 'ENVI-met 6.0 Albero: default root zone of a new grid plant and the shipped root range presets.'
SRC_MOSER = ('Moser-Reischl, A., Franceschi, E., Rahman, M.A. et al. (2025): Spatial and temporal dynamics of the '
             'leaf area index (LAI) of selected tree species in urban environments. '
             'Urban Forestry & Urban Greening 107: 128795.')

ROOT_PRESETS = {
    'Generic: cylinder (homogeneous)': {
        'settings': {'shape': roots.SHAPE_CYLINDER, 'distribution': roots.DIST_UNIFORM},
        'source': 'Geometric form without species reference.',
    },
    'Generic: half ellipsoid': {
        'settings': {'shape': roots.SHAPE_ELLIPSOID, 'distribution': roots.DIST_LINEAR},
        'source': 'Geometric form without species reference.',
    },
    'Generic: heart root system': {
        'settings': {'shape': roots.SHAPE_HEART, 'distribution': roots.DIST_LINEAR},
        'source': 'Geometric form without species reference.',
    },
    'Generic: tap root system': {
        'settings': {'shape': roots.SHAPE_TAP, 'distribution': roots.DIST_LINEAR},
        'source': 'Geometric form without species reference.',
    },
    'Generic: plate (flat) root system': {
        'settings': {'shape': roots.SHAPE_PLATE, 'distribution': roots.DIST_UNIFORM, 'depth': 0.8},
        'source': 'Geometric form without species reference; the depth of 0.8 m is only a starting value.',
    },
    'ENVI-met default (new Albero tree)': {
        'settings': {'shape': roots.SHAPE_CYLINDER, 'distribution': roots.DIST_UNIFORM, 'depth': 4.0,
                     'size_mode': 'diameter', 'diameter': 10.0, 'density_mode': 'rad', 'rad': 0.1},
        'source': SRC_ENVIMET,
    },
    'Deciduous trees, temperate (Jackson et al. 1996)': {
        'settings': {'distribution': roots.DIST_EXPONENTIAL, 'beta': 0.966},
        'source': 'Vertical distribution Y = 1 - beta^d (d in cm), beta = 0.966 for temperate deciduous forest '
                  '(65 % of the roots in the upper 30 cm). ' + SRC_JACKSON + ' Model: ' + SRC_GALE,
    },
    'Conifers, temperate (Jackson et al. 1996)': {
        'settings': {'distribution': roots.DIST_EXPONENTIAL, 'beta': 0.976},
        'source': 'Vertical distribution Y = 1 - beta^d (d in cm), beta = 0.976 for temperate coniferous forest '
                  '(52 % of the roots in the upper 30 cm). ' + SRC_JACKSON + ' Model: ' + SRC_GALE,
    },
    'Forest, cool-temperate (Schenk & Jackson 2002)': {
        'settings': {'distribution': roots.DIST_LOGISTIC, 'd50': 21.0, 'd95': 104.0},
        'source': 'Logistic model r(D) = 1 / (1 + (D / D50)^c): 50 % of the roots above D50 = 21 cm, 95 % above '
                  'D95 = 104 cm (cool-temperate forest including plantations; conifers and broadleaved trees do not '
                  'differ in the underlying data set). ' + SRC_SCHENK,
    },
    'Forest, warm-temperate (Schenk & Jackson 2002)': {
        'settings': {'distribution': roots.DIST_LOGISTIC, 'd50': 23.0, 'd95': 121.0},
        'source': 'Logistic model, D50 = 23 cm, D95 = 121 cm (warm-temperate forest including plantations). '
                  + SRC_SCHENK,
    },
    'Forest, boreal (Schenk & Jackson 2002)': {
        'settings': {'distribution': roots.DIST_LOGISTIC, 'd50': 12.0, 'd95': 58.0},
        'source': 'Logistic model, D50 = 12 cm, D95 = 58 cm (boreal forest). ' + SRC_SCHENK,
    },
    'European beech (Fagus sylvatica): heart root, 1.25 m deep': {
        'settings': {'shape': roots.SHAPE_HEART, 'depth': 1.25, 'distribution': roots.DIST_LOGISTIC,
                     'd50': 21.0, 'd95': 104.0},
        'source': ROOT_TYPE_NOTE + ' ' + DEPTH_NOTE + ' ' + LOGISTIC_NOTE,
    },
    'Oak (Quercus robur, Q. petraea): tap root, 1.26 m deep': {
        'settings': {'shape': roots.SHAPE_TAP, 'depth': 1.26, 'distribution': roots.DIST_LOGISTIC,
                     'd50': 21.0, 'd95': 104.0},
        'source': ROOT_TYPE_NOTE + ' ' + DEPTH_NOTE + ' ' + LOGISTIC_NOTE,
    },
    'Scots pine (Pinus sylvestris): tap root, 1.47 m deep': {
        'settings': {'shape': roots.SHAPE_TAP, 'depth': 1.47, 'distribution': roots.DIST_LOGISTIC,
                     'd50': 21.0, 'd95': 104.0},
        'source': ROOT_TYPE_NOTE + ' ' + DEPTH_NOTE + ' ' + LOGISTIC_NOTE,
    },
    'Norway spruce (Picea abies): surface root system, 0.96 m deep': {
        'settings': {'shape': roots.SHAPE_PLATE, 'depth': 0.96, 'distribution': roots.DIST_LOGISTIC,
                     'd50': 21.0, 'd95': 104.0},
        'source': ROOT_TYPE_NOTE + ' ' + DEPTH_NOTE + ' ' + LOGISTIC_NOTE,
    },
    'Lime, Norway maple, birch, larch: heart root system (shape only)': {
        'settings': {'shape': roots.SHAPE_HEART},
        'source': ROOT_TYPE_NOTE + ' No checked depth value for these species; the depth stays as set.',
    },
    'Ash, aspen: surface root system (shape only)': {
        'settings': {'shape': roots.SHAPE_PLATE},
        'source': ROOT_TYPE_NOTE + ' No checked depth value for these species; the depth stays as set.',
    },
    'Woody plants: root system 2.3 x crown width (Tumber-Davila et al. 2022)': {
        'settings': {'size_mode': 'factor', 'radius_factor': 2.3},
        'source': 'Mean ratio of shoot width to root system width of woody plants is 0.44 (individual plants '
                  'worldwide, hardly changing with aridity), i.e. root systems are on average about 2.3 times as '
                  'wide as the crown. ' + SRC_TUMBER,
    },
    'Young open-grown trees: root spread 2.9 x crown (Gilman 1988)': {
        'settings': {'size_mode': 'factor', 'radius_factor': 2.9},
        'source': 'Root system diameter of young nursery trees averaged 2.9 times the canopy diameter, from 1.68 '
                  '(Fraxinus pennsylvanica) to 3.77 (Magnolia grandiflora); trunk diameters below 18 cm, so the '
                  'ratio may not hold for mature trees, and paved urban sites restrict the spread. ' + SRC_GILMAN +
                  ' Values as reported in ' + SRC_DAY,
    },
}

# Orientation values only (no hard limits): maximum summer LAI of urban trees.
LAI_REFERENCES = {
    'Tilia cordata': {'lai_max': 4.7, 'note': 'highest of 15 urban species, n = 341', 'source': SRC_MOSER},
    'Acer platanoides': {'lai_max': 3.4, 'note': 'n = 62', 'source': SRC_MOSER},
    'Gleditsia triacanthos': {'lai_max': 2.4, 'note': 'lowest of 15 urban species', 'source': SRC_MOSER},
}


def root_preset_names():
    return list(ROOT_PRESETS)

def lai_reference_text():
    # One line per species for tooltips.
    return '\n'.join('%s: max. LAI %.1f (%s)' % (name, v['lai_max'], v['note'])
                     for name, v in LAI_REFERENCES.items())
