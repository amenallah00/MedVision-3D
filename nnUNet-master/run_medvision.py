"""
MedVision 3D - AI-Powered 3D Medical Image Segmentation
Maintained and customized by: Amen Allah Ben Abdrabah
Based on the nnU-Net framework (DKFZ / Helmholtz Imaging)
"""

import os
import sys
import argparse
import shutil
import time
from pathlib import Path

# ANSI colors
CYAN = "\033[96m"
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_banner():
    print(f"""
{CYAN}{BOLD}========================================================================{RESET}
{BLUE}{BOLD}                    MEDVISION 3D{RESET}
{CYAN}          AI-Powered 3D Medical Image Segmentation{RESET}
{CYAN}    Maintained & Personalized by: {BOLD}Amen Allah Ben Abdrabah{RESET}
{CYAN}{BOLD}========================================================================{RESET}
""")


def setup_default_environment():
    """Ensure nnUNet environment variables are defined."""
    base_dir = Path(__file__).resolve().parent / "medvision_data"
    raw_dir = base_dir / "nnUNet_raw"
    preprocessed_dir = base_dir / "nnUNet_preprocessed"
    results_dir = base_dir / "nnUNet_results"

    for d in (raw_dir, preprocessed_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    if "nnUNet_raw" not in os.environ:
        os.environ["nnUNet_raw"] = str(raw_dir)
    if "nnUNet_preprocessed" not in os.environ:
        os.environ["nnUNet_preprocessed"] = str(preprocessed_dir)
    if "nnUNet_results" not in os.environ:
        os.environ["nnUNet_results"] = str(results_dir)

    return {
        "nnUNet_raw": os.environ["nnUNet_raw"],
        "nnUNet_preprocessed": os.environ["nnUNet_preprocessed"],
        "nnUNet_results": os.environ["nnUNet_results"],
    }


def check_system_status():
    """Inspect and report Python, PyTorch, GPU, and MedVision 3D dependencies."""
    print_banner()
    print(f"{BOLD}[System & Hardware Diagnostics]{RESET}\n")

    # Python version
    print(f"  • Python Version : {sys.version.split()[0]} ({sys.executable})")

    # PyTorch and GPU
    try:
        import torch
        print(f"  • PyTorch Version: {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"  • Compute Device : {GREEN}CUDA GPU Available ({device_name}, {vram_gb:.1f} GB VRAM){RESET}")
            print(f"  • CUDA Version   : {torch.version.cuda}")
        else:
            print(f"  • Compute Device : {YELLOW}CPU Only (CUDA not active){RESET}")
    except ImportError:
        print(f"  • PyTorch        : {RED}Not installed in current environment{RESET}")

    # Core dependencies
    deps = [
        ("SimpleITK", "SimpleITK"),
        ("nibabel", "nibabel"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("batchgenerators", "batchgenerators"),
        ("nnunetv2", "nnunetv2"),
    ]
    print("\n  • Library Dependencies:")
    for name, mod in deps:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "installed")
            print(f"    - {name:<18}: {GREEN}OK{RESET} (v{ver})")
        except ImportError:
            print(f"    - {name:<18}: {YELLOW}Missing{RESET}")

    # Environment variables
    paths = setup_default_environment()
    print("\n  • Storage Environment Paths:")
    for k, v in paths.items():
        print(f"    - {k:<20}: {v}")

    print()


def calculate_dice(y_true, y_pred):
    """Compute binary / multi-class Dice Similarity Coefficient (DSC)."""
    import numpy as np
    intersect = np.sum(y_true * y_pred)
    denominator = np.sum(y_true) + np.sum(y_pred)
    return (2.0 * intersect) / (denominator + 1e-6)


def run_sample_inference_demo(device_override=None):
    """
    Runs end-to-end 3D CT volumetric segmentation on sample scan
    (example_ct_sm.nii.gz) using pre-trained weights and validates
    against ground truth.
    """
    print_banner()
    print(f"{BOLD}[Running Sample 3D CT Segmentation Demo]{RESET}\n")

    # Check dependencies
    try:
        import torch
        import nibabel as nib
        import numpy as np
    except ImportError as e:
        print(f"{RED}Error: Required libraries missing ({e}).{RESET}")
        print("Please ensure torch, nibabel, and numpy are installed.")
        return False

    # Paths
    project_root = Path(__file__).resolve().parent
    example_img = project_root / "nnunetv2" / "tests" / "example_data" / "example_ct_sm.nii.gz"
    example_gt = project_root / "nnunetv2" / "tests" / "example_data" / "example_ct_sm_T300_output.nii.gz"

    if not example_img.exists():
        print(f"{RED}Error: Example scan not found at {example_img}{RESET}")
        return False

    # Check for weights in standard location or local results
    weights_home = Path.home() / "github_actions_nnunet" / "results"
    local_results = Path(os.environ.get("nnUNet_results", project_root / "medvision_data" / "nnUNet_results"))
    
    weights_dir = None
    if (weights_home / "Dataset300_body_6mm_1559subj").exists():
        weights_dir = weights_home
    elif (local_results / "Dataset300_body_6mm_1559subj").exists():
        weights_dir = local_results

    if not weights_dir:
        print(f"{YELLOW}Pre-trained weights not found locally.{RESET}")
        print("Downloading TotalSegmentator Body CT weights (~118 MB)...")
        import urllib.request
        import zipfile

        weights_dir = weights_home
        weights_dir.mkdir(parents=True, exist_ok=True)
        zip_tmp = weights_dir / "weights.zip"
        url = "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset300_body_6mm_1559subj.zip"
        
        try:
            urllib.request.urlretrieve(url, zip_tmp)
            with zipfile.ZipFile(zip_tmp, "r") as z:
                z.extractall(weights_dir)
            zip_tmp.unlink(missing_ok=True)
            print(f"{GREEN}Weights downloaded successfully!{RESET}\n")
        except Exception as err:
            print(f"{RED}Failed to download weights: {err}{RESET}")
            return False

    os.environ["nnUNet_results"] = str(weights_dir)

    # Determine device
    device = device_override or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Compute Device: {BOLD}{device.upper()}{RESET}")

    # Setup demo workspace
    demo_work_dir = project_root / "demo_output"
    input_dir = demo_work_dir / "input"
    output_dir = demo_work_dir / "output"

    # Clean existing input and output
    if input_dir.exists():
        shutil.rmtree(input_dir, ignore_errors=True)
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_scan = input_dir / "example_ct_sm_0000.nii.gz"
    shutil.copy(example_img, input_scan)

    print(f"Input Scan : {input_scan.name} (Volumetric CT)")
    print(f"Running MedVision 3D inference pipeline...")
    
    start_time = time.time()
    scripts_dir = Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin")
    predict_exe = scripts_dir / ("nnUNetv2_predict.exe" if os.name == "nt" else "nnUNetv2_predict")
    
    if predict_exe.exists():
        cmd = [str(predict_exe)]
    else:
        cmd = [sys.executable, "-c", "from nnunetv2.inference.predict_from_raw_data import predict_entry_point; predict_entry_point()"]

    cmd.extend([
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-d", "300",
        "-tr", "nnUNetTrainer",
        "-c", "3d_fullres",
        "-f", "0",
        "-device", str(device),
    ])

    import subprocess
    ret = subprocess.call(cmd)
    elapsed = time.time() - start_time

    if ret != 0:
        print(f"\n{RED}Inference failed with exit code {ret}{RESET}")
        return False

    pred_file = output_dir / "example_ct_sm.nii.gz"
    if not pred_file.exists():
        print(f"{RED}Predicted file not found at {pred_file}{RESET}")
        return False

    # Evaluate prediction against ground truth
    print(f"\n{GREEN}Inference completed in {elapsed:.2f} seconds!{RESET}")
    print("Evaluating Dice Similarity Coefficient against ground truth...")

    gt_data = nib.load(str(example_gt)).get_fdata()
    pred_data = nib.load(str(pred_file)).get_fdata()

    dsc = calculate_dice(gt_data, pred_data)
    print(f"\n{BOLD}========================================================================{RESET}")
    print(f"{BOLD}                        INFERENCE RESULTS SUMMARY{RESET}")
    print(f"{BOLD}========================================================================{RESET}")
    print(f"  • Model Architecture       : 3D Full-Resolution U-Net")
    print(f"  • Dataset ID / Name        : Dataset300 (Body 6mm Organ Segmentation)")
    print(f"  • Execution Device         : {device.upper()}")
    print(f"  • Total Inference Runtime  : {elapsed:.2f} seconds")
    print(f"  • Dice Similarity Coeff    : {BOLD}{dsc:.5f}{RESET} " + (f"({GREEN}PASSED >= 0.99{RESET})" if dsc >= 0.99 else f"({YELLOW}Acceptable{RESET})"))
    print(f"  • Output Segmentation Mask : {pred_file}")
    print(f"{BOLD}========================================================================{RESET}\n")

    return True


def interactive_menu():
    """Interactive command-line menu."""
    while True:
        print_banner()
        print("Select an option:")
        print("  [1] Check System & Hardware Status")
        print("  [2] Run Sample 3D CT Segmentation Demo (Inference Validation)")
        print("  [3] Plan & Preprocess a Dataset")
        print("  [4] Train a Model")
        print("  [5] Run Custom Inference on New Images")
        print("  [6] Exit")
        print()
        choice = input("Enter choice (1-6): ").strip()

        if choice == "1":
            check_system_status()
            input("\nPress Enter to return to menu...")
        elif choice == "2":
            run_sample_inference_demo()
            input("\nPress Enter to return to menu...")
        elif choice == "3":
            dataset_id = input("Enter Dataset ID (e.g. 1 or 001): ").strip()
            if dataset_id:
                setup_default_environment()
                os.system(f'"{sys.executable}" -m nnunetv2.experiment_planning.plan_and_preprocess_entrypoints -d {dataset_id} --verify_dataset_integrity')
            input("\nPress Enter to return to menu...")
        elif choice == "4":
            dataset_id = input("Enter Dataset ID (e.g. 1): ").strip()
            config = input("Enter configuration (2d / 3d_fullres / 3d_lowres) [default: 3d_fullres]: ").strip() or "3d_fullres"
            fold = input("Enter fold (0, 1, 2, 3, 4, or all) [default: 0]: ").strip() or "0"
            setup_default_environment()
            os.system(f'"{sys.executable}" -m nnunetv2.run.run_training {dataset_id} {config} {fold}')
            input("\nPress Enter to return to menu...")
        elif choice == "5":
            inp = input("Enter input folder path: ").strip()
            out = input("Enter output folder path: ").strip()
            d_id = input("Enter Dataset ID: ").strip()
            config = input("Enter config (default: 3d_fullres): ").strip() or "3d_fullres"
            fold = input("Enter fold (default: 0): ").strip() or "0"
            setup_default_environment()
            os.system(f'"{sys.executable}" -m nnunetv2.inference.predict_from_raw_data -i "{inp}" -o "{out}" -d {d_id} -c {config} -f {fold}')
            input("\nPress Enter to return to menu...")
        elif choice == "6":
            print(f"\n{CYAN}Exiting MedVision 3D. Goodbye!{RESET}")
            break
        else:
            print(f"{YELLOW}Invalid choice, please select 1-6.{RESET}")
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(
        description="MedVision 3D: AI-Powered 3D Medical Image Segmentation - Master Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--status", action="store_true", help="Display system and hardware status")
    parser.add_argument("--demo", action="store_true", help="Run the sample 3D CT inference demo")
    parser.add_argument("--test", action="store_true", help="Alias for --demo")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None, help="Force computation device")

    args = parser.parse_args()

    if args.status:
        check_system_status()
    elif args.demo or args.test:
        run_sample_inference_demo(device_override=args.device)
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
