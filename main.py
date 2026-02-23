from src.pipeline.l2arctic_minimal import run_l2arctic_minimal
from src.pipeline.saa_minimal import run_saa_minimal
from src.data.saa_utils import load_saa_samples


if __name__ == "__main__":
    #run_l2arctic_minimal()
    samples = load_saa_samples("data/raw/archive.zip")[:20]
    run_saa_minimal(samples=samples)
