#!/bin/sh
# Container entry point: bring the database schema up to date, then serve. Exits (and the platform restarts
# the container) if the database can't be reached or a migration fails.
set -e
cd /app
alembic upgrade head
mkdir -p /data/run && cd /data/run   # relative paths (.kartrix/logs, traces) land on /data
exec kartrix serve
