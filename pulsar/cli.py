from __future__ import annotations

import argparse

from pulsar.config import Settings
from pulsar.db import Database
from pulsar.security import new_api_key


def main() -> None:
    parser = argparse.ArgumentParser(prog="pulsar", description="Pulsar AI command line")
    sub = parser.add_subparsers(dest="command", required=True)

    key = sub.add_parser("key", help="API key commands")
    key_sub = key.add_subparsers(dest="key_command", required=True)
    create = key_sub.add_parser("create", help="Create an API key")
    create.add_argument("--name", required=True)
    create.add_argument("--daily-limit", type=int, default=1000)
    create.add_argument("--permissions", default="chat")

    args = parser.parse_args()
    settings = Settings()
    db = Database(settings.db_path)
    db.init()

    if args.command == "key" and args.key_command == "create":
        permissions = [p.strip() for p in args.permissions.split(",") if p.strip()]
        raw, record = new_api_key(args.name, permissions, args.daily_limit)
        db.insert_key(record)
        print("Pulsar API key created. This raw key is shown once:")
        print(raw)
        print(f"id={record['id']} prefix={record['prefix']} daily_limit={record['daily_limit']}")


if __name__ == "__main__":
    main()
