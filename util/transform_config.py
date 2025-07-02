#!/usr/bin/env python3
import json
import sys


def transform_config(input_file, output_file="config.json"):
    with open(input_file, "r") as f:
        service_key = json.load(f)

    config = {
        "AICORE_AUTH_URL": service_key["url"],
        "AICORE_CLIENT_ID": service_key["clientid"],
        "AICORE_CLIENT_SECRET": service_key["clientsecret"],
        "AICORE_RESOURCE_GROUP": "default",
        "AICORE_BASE_URL": service_key["serviceurls"]["AI_API_URL"],
    }

    with open(output_file, "w") as f:
        json.dump(config, f, indent=1)

    print(f"Config successfully written to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python transform_config.py <input_service_key.json> [output_config.json]"
        )
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "config.json"
    transform_config(input_file, output_file)
    print(
        "Done. Please confirm the file and move it to ~/.aicore/config.json for the workshop"
    )
