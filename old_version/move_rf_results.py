from pathlib import Path
import shutil

def move_results():
    # Source and destination paths
    src_base = Path("src/models/forecasting/results/neuralprophet")
    dst_base = Path("results/neuralprophet")
    
    # Create destination directory if it doesn't exist
    dst_base.mkdir(parents=True, exist_ok=True)
    
    # Find all run directories
    run_dirs = [d for d in src_base.glob("run_*") if d.is_dir()]
    
    if not run_dirs:
        print("No run directories found in the source location.")
        return
    
    # Move each run directory
    for src_dir in run_dirs:
        run_name = src_dir.name
        dst_dir = dst_base / run_name
        
        if dst_dir.exists():
            print(f"Skipping {run_name} - already exists in destination")
            continue
            
        print(f"Moving {run_name}...")
        shutil.move(str(src_dir), str(dst_dir))
    
    print("\nAll runs have been moved to:", dst_base.absolute())

if __name__ == "__main__":
    move_results()
