# cets-membrain

Converters between [MemBrain-seg](https://github.com/teamtomo/membrain-seg) membrane
segmentation outputs and CETS, the cryo-electron tomography metadata standard defined in
[cets-data-models](https://github.com/TomoBabel/cets-data-models).

| Data | MemBrain-seg files | CETS objects | Directions |
|---|---|---|---|
| Segmentation | `<tomo>[_<checkpoint>]_segmented.mrc` (binary, or connected components with `--store-connected-components`) | `SegmentationMask3D` (`semantic` or `instance`) | MemBrain → CETS, CETS → MemBrain |
| Skeleton | `<segmentation>_skel.mrc` (`membrain skeletonize`) | `SegmentationMask3D` (`semantic`) | MemBrain → CETS, CETS → MemBrain |
| Score map | `<tomo>_scores.mrc` (`--store-probabilities`) | `ProbabilityMap3D` (`value_type: logit`) | MemBrain → CETS, CETS → MemBrain |

All of them are on the voxel grid of the segmented tomogram and are linked to it through
`source_tomogram_id`, with the membrane as their annotated object.

## Installation

Python 3.11 or newer. The CETS data model is not declared as a dependency, so install it first:

```bash
pip install -e /path/to/cets-data-models
pip install -e .          # or: pip install -e .[dev] for the development tools
```

This installs the `membrain` package and the `cets-membrain` command. MemBrain-seg itself is
not needed.

## Command line

```text
cets-membrain {segmentation,scores} {to-cets,from-cets} [options]
```

```bash
# Score map, its segmentation and the skeleton of the segmentation
cets-membrain scores to-cets --scores TS_03_scores.mrc --tomo-id TS_03 -o scores.yaml
cets-membrain segmentation to-cets --segmentation TS_03_segmented.mrc --tomo-id TS_03 \
    --derived-from membrain_scores_TS_03 -o seg.yaml
cets-membrain segmentation to-cets --segmentation TS_03_segmented_skel.mrc --tomo-id TS_03 \
    --derived-from membrain_binary_TS_03 -o skel.yaml

# CETS -> MemBrain-seg file names
cets-membrain segmentation from-cets -i seg.yaml --output-dir out [--checkpoint my_ckpt]
cets-membrain scores from-cets -i scores.yaml --output-dir out
```

`to-cets` writes a CETS yaml file with the entry `segmentation` or `scores`; `from-cets` reads
it. Every sub-command has `--help`. Errors print a one-line message and exit with code 1; put
`--debug` before the data type to also print the traceback.

| `to-cets` option | Meaning |
|---|---|
| `--segmentation` / `--scores` (required) | The mrc file. |
| `--tomo-id` (required) | Id of the CETS tomogram it was obtained from (`source_tomogram_id`). It also names the coordinate systems (`<tomo id>_array`, `<tomo id>_physical`), so it must be the id of the CETS tomogram: MemBrain-seg file names do not contain a reliable one. |
| `--pixel-size` | Sampling rate (Å). Default: the one of the mrc header, which MemBrain-seg copies from the tomogram; an error is raised if neither is known. |
| `--mask-type {binary,connected_components,skeleton}` | Kind of mask. Default: skeleton if the file ends in `_skel.mrc`, else binary or connected components depending on its highest label. |
| `--derived-from` | Id of the annotation it was computed from (`derived_from_annotation_id`): the score map of a segmentation, the segmentation of a skeleton. |
| `--threshold` / `--unknown-threshold` | `--segmentation-threshold` used to segment the scores (default 0.0, MemBrain-seg's), or not recorded. |
| `--annotation-id` | Id of the annotation, unique within its region (needed when several runs on the same tomogram are converted). |

| `from-cets` option | Meaning |
|---|---|
| `--output-dir` (required) | Directory to write the file in. |
| `--tomo-name` | Tomogram name in the file names (the segmented tomogram file name without extension). Default: the `source_tomogram_id`. |
| `--checkpoint` | Model checkpoint name, which MemBrain-seg adds to the segmentation names. |
| `--skeleton` / `--no-skeleton` | Whether the mask is a skeleton (`*_skel.mrc`). Default: whether its file ends in `_skel.mrc`. |
| `--copy` | Copy the file instead of linking it (it is always copied if its header sampling rate is not the CETS one). |

## Python API

```python
from membrain.converters.scores import MembrainScores
from membrain.converters.segmentation import MembrainSegmentation

scores = MembrainScores().membrain_to_cets("TS_03_scores.mrc", "TS_03")
mask = MembrainSegmentation().membrain_to_cets(
    "TS_03_segmented.mrc", "TS_03", derived_from_annotation_id=scores.id
)
MembrainSegmentation().cets_to_membrain(mask, "out", tomo_name="TS_03")
```

`membrain.utils.discovery.find_membrain_outputs` finds the MemBrain-seg outputs of a directory
and the tomogram and skeleton each of them comes from.

## Conventions

- **Values.** Binary segmentations and skeletons are `semantic` masks whose membrane label is 1
  (background 0). Connected-component masks are `instance` masks: each value 1..N is a
  membrane. The score map holds the raw network outputs (logits, unbounded; a voxel is
  membrane if its score is above the segmentation threshold), not probabilities.
- **Files are not converted.** CETS points to the MemBrain-seg mrc files; `from-cets` links
  them, or copies them when their header sampling rate is not the CETS one.

## Limitations

- The segmentation threshold and the model checkpoint are not recorded in the files, so they
  have to be given.
- Connected-component masks get no per-label annotated objects.
- The masks are assumed to be on the voxel grid of the tomogram given with `--tomo-id`; this is
  not checked.

## Development

```bash
pip install -e .[dev]
pre-commit install
pytest membrain/tests/tests_*.py  # the test files must be given: they are named tests_*.py
ruff check . && ruff format .
mypy membrain
```

### Tests

| File | What it tests |
|---|---|
| `membrain/tests/tests_cets_membrain.py` | MemBrain-seg → CETS: segmentations, skeletons and score maps. |
| `membrain/tests/tests_cets_to_membrain.py` | CETS → MemBrain-seg: the file names, links and copies. |
| `membrain/tests/tests_discovery.py` | Finding the MemBrain-seg outputs of a directory and the segmentation of each skeleton. |
| `membrain/tests/tests_cli.py` | Every `cets-membrain` sub-command, on small synthetic files (built in a temporary directory). |
| `membrain/tests/tests_cli_fixtures.py` | Every `cets-membrain` sub-command on the data of two real MemBrain-seg runs (`membrain/tests/data`): the outputs are compared with the values of the input files and with reference CETS files, plus the links between annotations, the options, the copies with another sampling rate and the errors. |

```bash
pytest membrain/tests/tests_cli_fixtures.py                    # one file
pytest membrain/tests/tests_cli_fixtures.py -k from_cets       # some tests
```

The test data (`membrain/tests/data`, `DATA_DIR` in `membrain/tests/__init__.py`) holds two
MemBrain-seg runs on the tomogram vTomo1:

| File | Content |
|---|---|
| `vTomo1.mrc` | The segmented tomogram (141 x 281 x 91 voxels). |
| `sampling_rate.txt` | Its sampling rate (1.0 Å/voxel), the one of all the mrc headers. |
| `segmentation_binary/vTomo1_scores.mrc` | Score map (logits). |
| `segmentation_binary/vTomo1_segmented.mrc` | Binary segmentation: the scores above 0.0. |
| `segmentation_connected_components_and_skel/vTomo1_scores.mrc` | Score map of the second run (the same values). |
| `segmentation_connected_components_and_skel/vTomo1_segmented.mrc` | The segmentation stored as connected components (17 membranes). |
| `segmentation_connected_components_and_skel/vTomo1_segmented_skel.mrc` | Skeleton of the segmentation. |
| `*/membrain_*_vTomo1_membrain_to_cets.yaml` | Reference CETS file of each mrc file of its directory. |

The reference CETS files hold what `to-cets` gives with `--tomo-id vTomo1` and the default
options, plus `--derived-from` linking each run's annotations: the segmentation to the score
map (`membrain_scores_vTomo1`) and the skeleton to the segmentation
(`membrain_connected_components_vTomo1`). Their paths are relative to `membrain/tests`, so
`from-cets` reads them from that directory (relative paths are resolved from the working
directory). If the conversion changes on purpose, regenerate them from `membrain/tests`:

```bash
cd membrain/tests
cets-membrain scores to-cets --scores data/segmentation_binary/vTomo1_scores.mrc --tomo-id vTomo1 \
    -o data/segmentation_binary/membrain_scores_vTomo1_membrain_to_cets.yaml
cets-membrain segmentation to-cets --segmentation data/segmentation_binary/vTomo1_segmented.mrc \
    --tomo-id vTomo1 --derived-from membrain_scores_vTomo1 \
    -o data/segmentation_binary/membrain_binary_vTomo1_membrain_to_cets.yaml
# ... the same for the three files of segmentation_connected_components_and_skel
sed -i "s#^  path: $PWD/data/#  path: ./data/#" data/*/*.yaml   # relative paths
```

and review their diff. The files in `membrain/tests/data` are never modified by the tests.
