# Solar Filament Segmentation Challenge 2026

This repository reproduces the Kaggle submission `Det533-E08-ROI04-SpatialQuality-Prune`. It scored **0.37** on the public leaderboard. Starting with the 180 official test JPEGs, [reproduce.ipynb](reproduce.ipynb) writes one row per detected filament in `output/submission.csv`.

## Run the notebook

1. Clone this repository with its `weights/` directory. Download and unpack the [official competition data](https://www.kaggle.com/competitions/filament-segmentation-2026/data). The input directory ends in `MAGFiLO_1.0_Kaggle_2026/test/test_images`.
2. Create a Python 3.11 virtual environment and activate it. From the repository root, run:

   ```console
   python -m pip install -r requirements.txt
   python -m ipykernel install --sys-prefix --name solar-filaments --display-name "Solar filaments"
   ```

3. Set `SOLAR_TEST_IMAGES` to the directory containing the original JPEGs. For example, in PowerShell use `$env:SOLAR_TEST_IMAGES = 'D:/data/MAGFiLO_1.0_Kaggle_2026/test/test_images'`. The notebook also recognizes the competition archive under the repository's `data/` directory and Kaggle's competition input mount.
4. Open `reproduce.ipynb`, select the **Solar filaments** kernel and run all cells. For the same run from a terminal:

   ```console
   jupyter execute reproduce.ipynb --kernel_name=solar-filaments --timeout=3600
   ```

Set `SOLAR_OUTPUT_CSV` to write the CSV elsewhere. The final cell verifies the header, row count and SHA-256 digest. The reference environment used CPython 3.11.9 on Windows, CUDA-enabled PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128 and an NVIDIA GeForce RTX 4080 Laptop GPU (12 GB, driver 617.42). The package versions are pinned in [requirements.txt](requirements.txt).

## How the masks are made

The [inference code](solar_filaments.py) loads the local YOLO11m-seg detector and predicts boxes on the full frame. A ResNet-18 encoder and U-Net decoder refine each box using grayscale intensity, a CLAHE image and a rectangular seed. The refiner averages four flip views and returns a probability map in the original 2,048 × 2,048 coordinates.

The main detections receive first claim on pixels. A second pass considers lower-scoring proposals after duplicate removal. A ridge model scores these extra masks and retains scores of at least 0.24. Four spatial quality heads score the accepted masks using their image crops, predicted probability maps, final shapes and eight scalar measurements. Their mean decides which instances enter the CSV. The [model definitions](filament_models.py), learned weights and [frozen scoring parameters](weights/inference.json) are included.

The final detector and refiner were fine-tuned on 533 official training observations. All 882 annotator versions stayed with their physical image. Training rotated through the versions, taking one set of independent instance outlines per image per epoch. The release uses detector epoch 8 and refiner epoch 4. A separate set of 73 observations supplied the development comparison. For the spatial heads, each observation received a development score from a head trained outside its time block, with a 14-day gap at the boundary. The resulting local PQ was 0.436867 across 112 annotator-image records. On the competition test images, the notebook averages the four trained heads.

## Reference output

| Check | Submitted prediction |
|---|---:|
| Observations processed | 180 |
| Observations with at least one predicted filament | 178 |
| Unique predicted instances | 1,262 |
| CSV columns | `filament_id`, `segmentation_rle` |

The encoder checks each compressed COCO RLE against the original predicted mask. The exporter reads the completed CSV back and checks that instance IDs are unique and masks within an observation are disjoint. Two observations have zero predictions. Their image IDs appear in the run receipt.

The submitted CSV has SHA-256 `a0324008edfd7a4ea14a990a03e27667269da93475d5c096729003f90844f93e`. Running the notebook from a clean copy on the reference machine reproduced every byte in 193 seconds, including the example image and final CSV checks. GPU kernels and library versions can affect borderline pixels. The [verification record](reproduction_check.json) lists the software versions and hashes of the released files used for this run.

For model and source credits, see [third-party notices](THIRD_PARTY.md).
