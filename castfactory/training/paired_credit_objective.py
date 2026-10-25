"""Lightweight scientific objective versions, separate from collection protocol."""

LEGACY_OBJECTIVE_VERSION = "paired-credit-v1"
POST_OBJECTIVE_VERSION = "paired-logistic-post-v2"
INTEGRATED_OBJECTIVE_VERSION = "paired-integrated-local-v3"
MAIN_T3_OBJECTIVE_VERSION = 'paired-main-pre4-t3-v4'
SEMANTIC_OBJECTIVE_VERSION = 'paired-main-pre4-semantic-v5'
NATURAL_OBJECTIVE_VERSION = 'paired-main-pre4-natural-v6'
BOUNDED_NATURAL_OBJECTIVE_VERSION = "paired-main-pre4-natural-bounded-v7"
NATURAL_OBJECTIVE_VERSIONS = (NATURAL_OBJECTIVE_VERSION, BOUNDED_NATURAL_OBJECTIVE_VERSION)
MAIN_T3_OBJECTIVE_VERSIONS = (MAIN_T3_OBJECTIVE_VERSION, SEMANTIC_OBJECTIVE_VERSION,
                            *NATURAL_OBJECTIVE_VERSIONS)
INTEGRATED_OBJECTIVE_VERSIONS = (INTEGRATED_OBJECTIVE_VERSION, *MAIN_T3_OBJECTIVE_VERSIONS)
INTEGRATED_COLLECTION_VERSION = "bounded-local-credit-r2-v1"
INTEGRATED_COLLECTION_V2 = "bounded-local-credit-r2-v2"
MAIN_ONLY_COLLECTION_VERSION = "main-only-credit-r2-v1"
NATURAL_COLLECTION_VERSION = "natural-main-credit-r3-v1"
COUNTERPART_PREFIX_COLLECTION_VERSION = "natural-main-credit-r4-prefix-v1"
NATURAL_COLLECTION_VERSIONS = (NATURAL_COLLECTION_VERSION, COUNTERPART_PREFIX_COLLECTION_VERSION)
COLLECTION_PROTOCOLS = (INTEGRATED_COLLECTION_VERSION, INTEGRATED_COLLECTION_V2,
                        MAIN_ONLY_COLLECTION_VERSION, *NATURAL_COLLECTION_VERSIONS)
INTEGRATED_SCHEMA_VERSION = "integrated-credit-v1"
OBJECTIVE_VERSIONS = (LEGACY_OBJECTIVE_VERSION, POST_OBJECTIVE_VERSION,
                      *INTEGRATED_OBJECTIVE_VERSIONS)


def normalize_objective_version(value=LEGACY_OBJECTIVE_VERSION):
    """Missing historical fields default to v1; invalid explicit values fail."""
    if value not in OBJECTIVE_VERSIONS:
        raise ValueError("Unsupported paired credit objective version: " + str(value))
    return value


def normalize_collection_protocol(value=INTEGRATED_COLLECTION_VERSION, *, objective_version=None):
    """Missing historical selectors retain the original literal protocol."""
    if value not in COLLECTION_PROTOCOLS:
        raise ValueError("Unsupported paired collection protocol: " + str(value))
    if objective_version in NATURAL_OBJECTIVE_VERSIONS and value not in NATURAL_COLLECTION_VERSIONS:
        raise ValueError("Natural T1 objective requires natural main collection")
    if value in NATURAL_COLLECTION_VERSIONS and objective_version not in NATURAL_OBJECTIVE_VERSIONS:
        raise ValueError("Natural main collection requires the natural T1 objective")
    if value == COUNTERPART_PREFIX_COLLECTION_VERSION and objective_version != BOUNDED_NATURAL_OBJECTIVE_VERSION:
        raise ValueError("Counterpart prefix collection requires bounded natural v7")
    if (objective_version in (MAIN_T3_OBJECTIVE_VERSION, SEMANTIC_OBJECTIVE_VERSION)
            and value != MAIN_ONLY_COLLECTION_VERSION):
        raise ValueError("Main pre4 T3 objective requires main-only collection")
    if (objective_version is not None and value != INTEGRATED_COLLECTION_VERSION
            and normalize_objective_version(objective_version) not in INTEGRATED_OBJECTIVE_VERSIONS):
        raise ValueError("New collection protocol requires the integrated credit objective")
    return value
