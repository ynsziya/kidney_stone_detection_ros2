from pathlib import Path

def main() -> None:
    root = Path(__file__).resolve().parent
    print("kidney stone ai is ready")
    print(f"project root: {root}")

    expected = [
        "app",
        "ai",
        "preprocessing",
        "postprocessing",
        "mesh",
        "visualization",
        "models",
        "data",
    ]
    missing = [name for name in expected if not (root / name).is_dir()]
    if missing:
        print(f"missing folders: {missing}")
    else:
        print("folder check: OK")

if __name__ == "__main__":
    main()