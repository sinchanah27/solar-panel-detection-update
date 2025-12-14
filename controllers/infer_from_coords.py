# controllers/infer_from_coords.py
from controllers.master_controller import load_config, run_pipeline

def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--submission_id", default="manual_run")
    parser.add_argument("--config", default="config/config.yaml")

    args = parser.parse_args()
    config = load_config(args.config)

    payload = {
        "latitude": args.lat,
        "longitude": args.lon,
        "submission_id": args.submission_id
    }

    result = run_pipeline(payload, config)
    print(result)

if __name__ == "__main__":
    main()
